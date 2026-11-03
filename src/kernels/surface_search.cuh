#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_SURFACE_SEARCH_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_SURFACE_SEARCH_CUH

#include "kernels/bounds.cuh"
#include "kernels/common.cuh"

namespace mrl {

// ---------------------------------------------------------------------------
// Hierarchical tile-pruned query-centric kernels
//
// The depth buffer maintains per-tile (kTileW x kTileH) min/max depth values
// (with a one-pixel halo) that are updated on insert. The tiled kernels below
// exploit them: instead of exhaustively scanning every pixel of a query's
// projected truncation-sphere AABB, warps first test a conservative lower
// bound of the distance from the query to ANY candidate inside a tile and
// skip the whole tile (512 pixels) if the bound exceeds the running best.
// The bound is the distance to the frustum-slab box spanned by the tile's
// pixel extents and its [min,max] depth range (corner products are exact for
// the bilinear pixel->camera map). Pruning uses a strict comparison against a
// slightly deflated bound (x0.9999), so candidates that could tie the final
// minimum are never skipped: exported distances are identical to the
// exhaustive scan.
// ---------------------------------------------------------------------------

// Scan one tile (clipped to the query AABB) for the closest surface sample.
// The warp cooperates lane-per-column (coalesced along u); lane 0 folds the
// tile minimum into the block-wide running best in shared memory.
__device__ __forceinline__ void scanTileClosest(
    const float* __restrict__ d_depth_buffer,
    size_t frame_px_off,
    int tu0,
    int tu1,
    int tv0,
    int tv1,
    AABB2D query_aabb,
    int lane,
    float query_x,
    float query_y,
    float query_z,
    float truncation_radius_sq,
    float cx,
    float cy,
    float inv_fx,
    float inv_fy,
    float z_near,
    float z_far,
    uint16_t num_cols,
    int v_first,
    int v_step,
    unsigned long long* s_best_packed,
    uint32_t* s_best_sq_bits,
    bool exact_patch) {
  const int su0 = max(static_cast<int>(query_aabb.u_min), tu0);
  const int su1 = min(static_cast<int>(query_aabb.u_max), tu1);
  const int sv0 = max(static_cast<int>(query_aabb.v_min), tv0);
  const int sv1 = min(static_cast<int>(query_aabb.v_max), tv1);
  uint64_t tile_best = INFINITY_PACKED;
  const int u = su0 + lane;
  if (u < su1) {
    for (int v = sv0 + v_first; v < sv1; v += v_step) {
      const float depth =
          d_depth_buffer[frame_px_off + static_cast<size_t>(v) * num_cols + u];
      if (depth < 0.0f) continue;
      float x_cam, y_cam;
      unprojectPixel(depth, u, v, cx, cy, inv_fx, inv_fy, x_cam, y_cam);
      float dx, dy;
      if (exact_patch) {
        // The pixel is the patch half a pixel either side of its ray at its
        // (min-pooled) depth: the nearest point is the query clamped into the
        // patch. A point sample on the ray over-estimates the distance to a
        // surface that fills the block by up to half a patch diagonal
        // (measured: +3 cm mean, +7 cm p90 at pool 7 next to trunks). With
        // this the result is a lower bound over the patch.
        const float hx = 0.5f * depth * inv_fx;
        const float hy = 0.5f * depth * inv_fy;
        dx = fminf(fmaxf(query_x, x_cam - hx), x_cam + hx) - query_x;
        dy = fminf(fmaxf(query_y, y_cam - hy), y_cam + hy) - query_y;
      } else {
        dx = x_cam - query_x;
        dy = y_cam - query_y;
      }
      const float dz = depth - query_z;
      float dist_sq = distSq3(dx, dy, dz);
      const bool is_truncated = dist_sq >= truncation_radius_sq;
      dist_sq = is_truncated ? truncation_radius_sq : dist_sq;
      const bool is_extremum = (depth == z_near) || (depth == z_far);
      const uint64_t packed = PackedResult(static_cast<uint16_t>(v),
                                           static_cast<uint16_t>(u),
                                           dist_sq,
                                           is_truncated,
                                           false,
                                           is_extremum)
                                  .val;
      if (packed < tile_best) tile_best = packed;
    }
  }
  for (int offset = NUM_THREADS_IN_WARP / 2; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(0xFFFFFFFF, tile_best, offset);
    if (other < tile_best) tile_best = other;
  }
  if (lane == 0 && tile_best != INFINITY_PACKED) {
    atomicMin(s_best_packed, static_cast<unsigned long long>(tile_best));
    // high 32 bits are the float bits of dist_sq (sign bit not set yet)
    atomicMin(s_best_sq_bits, static_cast<uint32_t>(tile_best >> 32));
  }
}

// Occlusion-boundary counterpart of scanTileClosest: identical per-pixel math
// to the exhaustive occlusion kernel (ray feet within the neighbor depth
// window, plus the near-plane candidate when active).

__global__ void closestPointQueryCentricTiledKernel(
    const float* __restrict__ d_depth_buffer,
    const float* __restrict__ d_min_block_depths,
    const float* __restrict__ d_max_block_depths,
    const float* __restrict__ d_min_macro_depths,
    const float* __restrict__ d_max_macro_depths,
    const Pixel* __restrict__ d_queries_frames_projected,
    size_t num_frames_queries,
    size_t num_queries,
    uint16_t num_rows,
    uint16_t num_cols,
    uint16_t tiles_x,
    uint16_t tiles_y,
    uint16_t macros_x,
    uint16_t macros_y,
    float cx,
    float cy,
    float fx,
    float fy,
    float z_near,
    float z_far,
    const float* __restrict__ d_frame_query_truncation,
    float truncation_radius,
    const float* __restrict__ d_queries_frames,
    const AABB2D* __restrict__ d_queries_aabbs,
    PackedResult* __restrict__ d_results,
    bool exact_patch) {
  __shared__ unsigned long long s_best_packed;  // min over the whole block
  __shared__ uint32_t s_best_sq_bits;           // float bits of best dist_sq
  __shared__ int s_next_tile;

  const size_t i_frame_query = blockIdx.x;
  if (i_frame_query >= num_frames_queries) return;
  const size_t i_frame = i_frame_query / num_queries;

  const float query_x = d_queries_frames[i_frame_query * 3 + 0];
  const float query_y = d_queries_frames[i_frame_query * 3 + 1];
  const float query_z = d_queries_frames[i_frame_query * 3 + 2];
  const AABB2D query_aabb = d_queries_aabbs[i_frame_query];
  const float local_truncation =
      getFrameQueryTruncation(d_frame_query_truncation,
                              i_frame_query,
                              truncation_radius);
  const float truncation_radius_sq = local_truncation * local_truncation;

  if (threadIdx.x == 0) {
    // same starting value the exhaustive path uses via initPackedResultsKernel
    s_best_packed = PackedResult::Truncated(truncation_radius_sq).val;
    s_best_sq_bits = __float_as_uint(truncation_radius_sq);
  }
  if (threadIdx.x == 0) s_next_tile = 0;
  __syncthreads();

  if (query_aabb.u_max > query_aabb.u_min &&
      query_aabb.v_max > query_aabb.v_min) {
    const int tx0 = query_aabb.u_min / kTileW;
    const int tx1 = (query_aabb.u_max - 1) / kTileW;
    const int ty0 = query_aabb.v_min / kTileH;
    const int ty1 = (query_aabb.v_max - 1) / kTileH;

    const float inv_fx = __frcp_rn(fx);
    const float inv_fy = __frcp_rn(fy);
    const int lane = threadIdx.x & (NUM_THREADS_IN_WARP - 1);
    const int warp = threadIdx.x / NUM_THREADS_IN_WARP;
    const int num_warps = blockDim.x / NUM_THREADS_IN_WARP;
    const size_t frame_px_off =
        i_frame * static_cast<size_t>(num_rows) * num_cols;
    const size_t frame_tile_off =
        i_frame * static_cast<size_t>(tiles_x) * tiles_y;
    const size_t frame_macro_off =
        i_frame * static_cast<size_t>(macros_x) * macros_y;

    // Probe: scan the tile containing the projected query first so the
    // running best is tight before the sweep (prunes most of the AABB).
    int probe_tx = -1;
    int probe_ty = -1;
    {
      const Pixel proj = d_queries_frames_projected[i_frame_query];
      if (proj.u >= 0 && proj.v >= 0) {
        const int ptx = proj.u / kTileW;
        const int pty = proj.v / kTileH;
        if (ptx >= tx0 && ptx <= tx1 && pty >= ty0 && pty <= ty1) {
          probe_tx = ptx;
          probe_ty = pty;
          {
            const int ptu0 = ptx * kTileW;
            const int ptu1 = min(ptu0 + kTileW, static_cast<int>(num_cols));
            const int ptv0 = pty * kTileH;
            const int ptv1 = min(ptv0 + kTileH, static_cast<int>(num_rows));
            // bound-check first: a probe tile that cannot beat the truncation
            // sentinel (e.g. all far plane) never needs scanning
            const size_t ptile_idx =
                frame_tile_off + static_cast<size_t>(pty) * tiles_x + ptx;
            const float pbound_sq =
                tileSlabBoundSq(query_x,
                                query_y,
                                query_z,
                                static_cast<float>(ptu0),
                                static_cast<float>(ptu1),
                                static_cast<float>(ptv0),
                                static_cast<float>(ptv1),
                                d_min_block_depths[ptile_idx],
                                d_max_block_depths[ptile_idx],
                                cx,
                                cy,
                                inv_fx,
                                inv_fy,
                                exact_patch ? 0.5f : 0.0f);
            if (pbound_sq <= truncation_radius_sq) {
              // Seed the running best from a strip around the projected row,
              // striped across warps. Any pixel subset is admissible here; the
              // sweep below still visits this tile in full.
              const int strip_v0 = max(ptv0, static_cast<int>(proj.v) - 2);
              const int strip_v1 = min(ptv1, static_cast<int>(proj.v) + 2);
              scanTileClosest(d_depth_buffer,
                              frame_px_off,
                              ptu0,
                              ptu1,
                              strip_v0,
                              strip_v1,
                              query_aabb,
                              lane,
                              query_x,
                              query_y,
                              query_z,
                              truncation_radius_sq,
                              cx,
                              cy,
                              inv_fx,
                              inv_fy,
                              z_near,
                              z_far,
                              num_cols,
                              warp,
                              num_warps,
                              &s_best_packed,
                              &s_best_sq_bits,
                              exact_patch);
            }
          }
        }
      }
    }
    __syncthreads();

    // Re-shrink the sweep domain: with the probe-tightened best, the sphere
    // that can still improve the result is usually much smaller than the
    // truncation sphere the AABB was built from. (Small pad keeps candidates
    // that tie the current best inside the domain.)
    int stx0 = tx0, stx1 = tx1, sty0 = ty0, sty1 = ty1;
    {
      const float best_sq_now = loadBestDistSq(&s_best_sq_bits);
      if (best_sq_now < truncation_radius_sq) {
        const float center[3] = {query_x, query_y, query_z};
        const float r_eff = sqrtf(best_sq_now) * 1.0002f;
        const AABB2D shrunk = computeSpheresProjectedAABB(center,
                                                          num_rows,
                                                          num_cols,
                                                          cx,
                                                          cy,
                                                          fx,
                                                          fy,
                                                          r_eff);
        const int su_min = max(static_cast<int>(query_aabb.u_min),
                               static_cast<int>(shrunk.u_min));
        const int su_max = min(static_cast<int>(query_aabb.u_max),
                               static_cast<int>(shrunk.u_max));
        const int sv_min = max(static_cast<int>(query_aabb.v_min),
                               static_cast<int>(shrunk.v_min));
        const int sv_max = min(static_cast<int>(query_aabb.v_max),
                               static_cast<int>(shrunk.v_max));
        if (su_max <= su_min || sv_max <= sv_min) {
          stx1 = stx0 - 1;  // empty: nothing can beat the probe result
        } else {
          stx0 = su_min / kTileW;
          stx1 = (su_max - 1) / kTileW;
          sty0 = sv_min / kTileH;
          sty1 = (sv_max - 1) / kTileH;
        }
      }
    }

    // Bound test for one tile, then scan it if it can still improve the
    // running best. The probe tile is re-tested here rather than skipped: the
    // probe only covers a strip of it. Comparisons are written as
    // !(bound <= best) so NaN and INF bounds prune, which also discards tiles
    // of never-written frames (their min depth is INF).
    auto process_tile = [&](int tx, int ty) {
      const size_t tile_idx =
          frame_tile_off + static_cast<size_t>(ty) * tiles_x + tx;
      const float dmin = d_min_block_depths[tile_idx];
      const float dmax = d_max_block_depths[tile_idx];
      // z-slab pre-reject: every candidate in the tile has z in [dmin, dmax],
      // so its distance is at least the z-gap; ~4 flops vs ~35 for the full
      // corner-product bound, and it catches most far/sky tiles
      const float z_gap = fmaxf(fmaxf(dmin - query_z, query_z - dmax), 0.0f);
      if (!(z_gap * z_gap * kTileBoundPad <= loadBestDistSq(&s_best_sq_bits))) {
        return;
      }
      const int tu0 = tx * kTileW;
      const int tu1 = min(tu0 + kTileW, static_cast<int>(num_cols));
      const int tv0 = ty * kTileH;
      const int tv1 = min(tv0 + kTileH, static_cast<int>(num_rows));
      const float bound_sq = tileSlabBoundSq(query_x,
                                             query_y,
                                             query_z,
                                             static_cast<float>(tu0),
                                             static_cast<float>(tu1),
                                             static_cast<float>(tv0),
                                             static_cast<float>(tv1),
                                             dmin,
                                             dmax,
                                             cx,
                                             cy,
                                             inv_fx,
                                             inv_fy,
                                             exact_patch ? 0.5f : 0.0f);
      if (!(bound_sq <= loadBestDistSq(&s_best_sq_bits))) return;
      scanTileClosest(d_depth_buffer,
                      frame_px_off,
                      tu0,
                      tu1,
                      tv0,
                      tv1,
                      query_aabb,
                      lane,
                      query_x,
                      query_y,
                      query_z,
                      truncation_radius_sq,
                      cx,
                      cy,
                      inv_fx,
                      inv_fy,
                      z_near,
                      z_far,
                      num_cols,
                      0,
                      1,
                      &s_best_packed,
                      &s_best_sq_bits,
                      exact_patch);
    };

    const int sntx = stx1 - stx0 + 1;
    const int num_sweep_tiles = sntx * (sty1 - sty0 + 1);
    if (num_sweep_tiles <= kFlatSweepTiles) {
      // small domain: the macro level cannot pay for itself
      // Tiles are handed out on demand rather than striped across the warps.
      // Their cost is wildly uneven under branch and bound: a bound test is a
      // few flops and a full scan is a few hundred, and the expensive ones
      // cluster around the query, so a static split leaves warps idle at the
      // block's last barrier.
      int t = -1;
      for (;;) {
        if (lane == 0) t = atomicAdd(&s_next_tile, 1);
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        if (t >= num_sweep_tiles) break;
        process_tile(stx0 + (t % sntx), sty0 + (t / sntx));
      }
    } else {
      // two-level sweep: macro tiles (8x8 tiles) first, then survivors
      const int mx0 = stx0 / kMacroTiles;
      const int mx1 = stx1 / kMacroTiles;
      const int my0 = sty0 / kMacroTiles;
      const int my1 = sty1 / kMacroTiles;
      const int nmx = mx1 - mx0 + 1;
      const int num_macros = nmx * (my1 - my0 + 1);

      // sweep outward from the projected pixel's macro tile: the winner is
      // usually near it, so the running best tightens early and prunes the
      // rest (processing order does not affect the argmin result)
      const int pmx = probe_tx >= 0 ? min(max(probe_tx / kMacroTiles, mx0), mx1)
                                    : (mx0 + mx1) / 2;
      const int pmy = probe_ty >= 0 ? min(max(probe_ty / kMacroTiles, my0), my1)
                                    : (my0 + my1) / 2;
      int m = -1;
      for (;;) {
        if (lane == 0) m = atomicAdd(&s_next_tile, 1);
        m = __shfl_sync(0xFFFFFFFF, m, 0);
        if (m >= num_macros) break;
        const int mx = outwardCoord(m % nmx, pmx, mx0, mx1);
        const int my = outwardCoord(m / nmx, pmy, my0, my1);
        const size_t macro_idx =
            frame_macro_off + static_cast<size_t>(my) * macros_x + mx;
        const float mmin = d_min_macro_depths[macro_idx];
        const float mmax = d_max_macro_depths[macro_idx];
        const float macro_z_gap =
            fmaxf(fmaxf(mmin - query_z, query_z - mmax), 0.0f);
        if (!(macro_z_gap * macro_z_gap * kTileBoundPad <=
              loadBestDistSq(&s_best_sq_bits))) {
          continue;
        }
        const int mu0 = mx * (kMacroTiles * kTileW);
        const int mu1 =
            min(mu0 + kMacroTiles * kTileW, static_cast<int>(num_cols));
        const int mv0 = my * (kMacroTiles * kTileH);
        const int mv1 =
            min(mv0 + kMacroTiles * kTileH, static_cast<int>(num_rows));
        const float macro_bound_sq = tileSlabBoundSq(query_x,
                                                     query_y,
                                                     query_z,
                                                     static_cast<float>(mu0),
                                                     static_cast<float>(mu1),
                                                     static_cast<float>(mv0),
                                                     static_cast<float>(mv1),
                                                     mmin,
                                                     mmax,
                                                     cx,
                                                     cy,
                                                     inv_fx,
                                                     inv_fy,
                                                     exact_patch ? 0.5f : 0.0f);
        if (!(macro_bound_sq <= loadBestDistSq(&s_best_sq_bits))) continue;

        const int ttx0 = max(stx0, mx * kMacroTiles);
        const int ttx1 = min(stx1, mx * kMacroTiles + kMacroTiles - 1);
        const int tty0 = max(sty0, my * kMacroTiles);
        const int tty1 = min(sty1, my * kMacroTiles + kMacroTiles - 1);
        for (int ty = tty0; ty <= tty1; ++ty) {
          for (int tx = ttx0; tx <= ttx1; ++tx) {
            process_tile(tx, ty);
          }
        }
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    // exactly one block owns this entry: plain store replaces the
    // init-kernel + global atomicMin of the exhaustive path
    d_results[i_frame_query].val = s_best_packed;
  }
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_SURFACE_SEARCH_CUH
