// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// DEVICE ONLY, gfx125x only.
//
// Adaptive PULL-only combine body. Differs from EpCombine1250xBody in:
//   1. XDB split: remote store fires before pre-scan; wait after
//   2. Pre-scan atomicMax for dynamic chunk sizing (overlaps XGMI propagation)
//   3. Staggered-load + block-wide sequential reduce
//   4. No QUAD path — PULL only
//   5. Drops __threadfence_block() after s_wait_tensorcnt (same-wavefront LDS)

#pragma once

#include "src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---- XDB split helpers (same logic as EpCrossDeviceBarrier1250x) ----------

template <EpCfg kCfg>
__device__ __forceinline__ void EpXdbSignal(EpArgs args, bool needGridRendezvous,
                                            unsigned long long& outPhase,
                                            unsigned int*& outFanLines, unsigned int& outFanEpoch) {
  constexpr int npes = kCfg.worldSize;
  const int thdId = threadIdx.x;
  const int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long win = args.window;

  const unsigned long long phase = args.xdbFlag[blockIdx.x];
  outPhase = phase;

  if (needGridRendezvous) {
    if (thdId == 0) atomicAdd(args.gridBarrier, 1u);
    if constexpr (!EpIsWideEp(kCfg)) {
      if (globalThdId < npes) {
        EpWaitEq(args.gridBarrier, static_cast<unsigned int>(gridDim.x));
        __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      if (thdId == 0) {
        EpWaitEq(args.gridBarrier, static_cast<unsigned int>(gridDim.x));
        __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
    }
  }

  if (globalThdId < npes) {
    if (needGridRendezvous) __threadfence_system();
    __hip_atomic_store(EpPeer<unsigned long long>(win, globalThdId, args.offXdb) + args.rank, phase,
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (thdId == 0) args.xdbFlag[blockIdx.x] = phase + 1;
  if (blockIdx.x == 0) {
    for (int b = (int)gridDim.x + thdId; b < EpXdbFlagSlots; b += (int)blockDim.x)
      args.xdbFlag[b] = phase + 1;
  }

  outFanLines = reinterpret_cast<unsigned int*>(args.combineBarrierFan);
  outFanEpoch = static_cast<unsigned int>(phase);
}

template <EpCfg kCfg>
__device__ __forceinline__ void EpXdbWait(EpArgs args, unsigned long long phase,
                                          unsigned int* fanLines, unsigned int fanEpoch) {
  constexpr int npes = kCfg.worldSize;
  const int thdId = threadIdx.x;
  const unsigned long long win = args.window;

  if (blockIdx.x == 0) {
    if (thdId < npes) {
      unsigned long long* slot = EpLocal<unsigned long long>(win, args.offXdb) + thdId;
      while (__hip_atomic_load(slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < phase)
        __builtin_amdgcn_s_sleep(MORI_COMB_BARSLEEP);
    }
    __syncthreads();
    __threadfence();
    for (int b = thdId; b < (int)gridDim.x; b += (int)blockDim.x)
      __hip_atomic_store(fanLines + (size_t)b * MORI_COMB_BARSPREAD, fanEpoch, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  } else {
    if (thdId == 0) {
      while (__hip_atomic_load(fanLines + (size_t)blockIdx.x * MORI_COMB_BARSPREAD,
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != fanEpoch)
        __builtin_amdgcn_s_sleep(MORI_COMB_BARSLEEP);
    }
    __syncthreads();
  }
  __syncthreads();
}

// ---- LDS tail: last 128 B reserved for metadata ---------------------------
// Tile budget is reduced by 128 B; the tail holds the work-stealing counter
// and the pre-scanned adaptMaxSrc.  Both are int-sized and never overlap with
// the ptr-array / tile region that occupies [0, budget-128).

constexpr size_t _cAdaptLdsTail = 128;
constexpr size_t _cAdaptTileBudget = MORI_COMB_LDS_BUDGET - _cAdaptLdsTail;

struct _AdaptMeta {
  int adaptMaxSrc;
  int wsCounter;
};
static_assert(sizeof(_AdaptMeta) <= _cAdaptLdsTail);

// ---- Main body ------------------------------------------------------------

template <EpCfg kCfg, typename T>
__device__ void EpCombine1250xAdaptBody(EpArgs args) {
  using TokT = T;
  constexpr int npes = kCfg.worldSize;
  constexpr int topk = kCfg.numExpertPerToken;
  constexpr int WS = kCfg.waveSize;

  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (WS - 1);
  const int warpId = thdId / WS;
  const int warpNum = kCfg.warpPerBlock;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = (int)gridDim.x * warpNum;
  const unsigned long long win = args.window;

  const index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  const size_t hiddenDim = (size_t)kCfg.hiddenDim;

  extern __shared__ char sharedMem[];
  _AdaptMeta& meta = *reinterpret_cast<_AdaptMeta*>(sharedMem + _cAdaptTileBudget);

  // ==================================================================
  // Phase 1: Staging copy
  // ==================================================================
  T* const stage = EpLocal<T>(win, args.offOutTok);
  bool staged = false;
  if (reinterpret_cast<const T*>(args.inpTokenBuf) != stage) {
    for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
      core::WarpCopy(stage + i * hiddenDim,
                     reinterpret_cast<const T*>(args.inpTokenBuf) + i * hiddenDim, hiddenDim);
    }
    staged = true;
  }
  if (staged) {
    __syncthreads();
    if (warpId == 0) __threadfence_system();
  }

  // ==================================================================
  // Phase 2: XDB signal — remote store out ASAP
  // ==================================================================
  unsigned long long _xdbPhase;
  unsigned int* _xdbFanLines;
  unsigned int _xdbFanEpoch;
  EpXdbSignal<kCfg>(args, staged, _xdbPhase, _xdbFanLines, _xdbFanEpoch);

  // ==================================================================
  // Phase 3: Pre-scan max source count (overlaps XGMI propagation)
  //   Only computes block-wide max — no per-token storage.
  // ==================================================================
  const int _numTokens = (int)args.numTokens;

  if (thdId == 0) meta.adaptMaxSrc = 0;
  __syncthreads();

  int _myMaxSrc = 0;
  for (int i = globalWarpId; i < _numTokens; i += globalWarpNum) {
    int nSrc = 0;
    for (int j = laneId; j < topk; j += WS) {
      index_t destTokId = args.dispDestTokIdMap[i * topk + j];
      index_t destPe = EpPeFromFlat<kCfg>(destTokId);
      if (destPe < npes) nSrc++;
    }
    for (int s = WS / 2; s > 0; s >>= 1) nSrc += __shfl_down(nSrc, s);
    if (laneId == 0) _myMaxSrc = max(_myMaxSrc, nSrc);
  }
  if (laneId == 0 && _myMaxSrc > 0) atomicMax(&meta.adaptMaxSrc, _myMaxSrc);

  // ==================================================================
  // Phase 4: XDB wait (final __syncthreads makes adaptMaxSrc visible)
  // ==================================================================
  EpXdbWait<kCfg>(args, _xdbPhase, _xdbFanLines, _xdbFanEpoch);
  if (globalWarpId == 0 && laneId == 0) *args.totalRecvTokenNum = 0;
  if (args.numTokens == 0) return;

  // ==================================================================
  // Dynamic tile sizing — per-warp budget
  // ==================================================================
  constexpr bool _cPullType = (sizeof(TokT) == 2 || sizeof(TokT) == 4);
  const int _cPullRowElems = 128 / (int)sizeof(TokT);
  const int _cPullSrcMaxStatic = (npes <= 4 && npes < topk) ? npes : topk;

  int _dynSrcMax = meta.adaptMaxSrc;
  if (_dynSrcMax > _cPullSrcMaxStatic) _dynSrcMax = _cPullSrcMaxStatic;
  if (_dynSrcMax < 1) _dynSrcMax = 1;

  const size_t _perWarpBudget = _cAdaptTileBudget / (size_t)warpNum;
  int _dynTileElems = (int)(_perWarpBudget / ((size_t)_dynSrcMax * sizeof(TokT)));
  _dynTileElems = (_dynTileElems / _cPullRowElems) * _cPullRowElems;

  const bool _dynPullOk =
      _cPullType && ((int)hiddenDim >= _cPullRowElems) && (_dynTileElems >= _cPullRowElems);

  TokT* _cPullTiles = nullptr;
  if constexpr (_cPullType) {
    _cPullTiles = reinterpret_cast<TokT*>(sharedMem) + (size_t)warpId * _dynSrcMax * _dynTileElems;
  }

  // Per-warp nSrc broadcast in LDS tail (after _AdaptMeta)
  int* _warpNSrcLds = reinterpret_cast<int*>(&meta + 1);

  // ==================================================================
  // Staggered-load + block-wide sequential reduce
  // ==================================================================
  const int _tokPerBlock = _numTokens / (int)gridDim.x;
  const int _tokRem = _numTokens % (int)gridDim.x;
  const int _tokStart = (int)blockIdx.x * _tokPerBlock + min((int)blockIdx.x, _tokRem);
  const int _tokCount = _tokPerBlock + ((int)blockIdx.x < _tokRem ? 1 : 0);

  for (int _batchOff = 0; _batchOff < _tokCount; _batchOff += warpNum) {
    const int _mySlot = _batchOff + warpId;
    const bool _myValid = (_mySlot < _tokCount);
    const int _myTokenId = _myValid ? (_tokStart + _mySlot) : 0;

    // Each warp computes source pointers for its own token
    TokT* _srcPtrs[topk];
    [[maybe_unused]] float* _srcWtPtrs[topk];
    int _myNSrc = 0;
    for (int j = 0; j < topk; ++j) {
      if (_myValid) {
        index_t destTokId = args.dispDestTokIdMap[_myTokenId * topk + j];
        index_t destPe = EpPeFromFlat<kCfg>(destTokId);
        if (destPe < npes) {
          index_t destLocalTokId = EpLocalTokFromFlat<kCfg>(destTokId);
          _srcPtrs[j] = EpPeer<TokT>(win, destPe, args.offOutTok) + destLocalTokId * hiddenDim;
          if constexpr (kCfg.useWeights)
            _srcWtPtrs[j] = EpPeer<float>(win, destPe, args.offOutWts) + destLocalTokId * topk;
          ++_myNSrc;
        } else {
          _srcPtrs[j] = nullptr;
          if constexpr (kCfg.useWeights) _srcWtPtrs[j] = nullptr;
        }
      } else {
        _srcPtrs[j] = nullptr;
        if constexpr (kCfg.useWeights) _srcWtPtrs[j] = nullptr;
      }
    }

    // Broadcast nSrc to all warps via LDS
    if (laneId == 0) _warpNSrcLds[warpId] = _myValid ? _myNSrc : 0;
    __syncthreads();

    // Per tile chunk
    for (size_t _off = 0; _off < hiddenDim; _off += _dynTileElems) {
      int _n = (int)(hiddenDim - _off);
      if (_n > _dynTileElems) _n = _dynTileElems;
      const bool _tdmOk = _dynPullOk && ((size_t)_n * sizeof(TokT) >= 128);

      // ---- Staggered TDM issue: warp 0..3 first ----
      if constexpr (_cPullType) {
        if (_tdmOk && _myValid && _myNSrc > 0) {
          const gfx1250_TDM_GROUP1 _pg1 = TdmShape<TokT>(_n);
          if (warpId < 4) {
            int _tileJ = 0;
            for (int _j = 0; _j < topk; ++_j) {
              if (_srcPtrs[_j] == nullptr) continue;
              TdmIssueLoad<TokT>(_cPullTiles + (size_t)_tileJ * _dynTileElems, _srcPtrs[_j] + _off,
                                 _pg1);
              ++_tileJ;
            }
          }
        }
      }
      __syncthreads();

      // ---- Warp 4..7 issue ----
      if constexpr (_cPullType) {
        if (_tdmOk && _myValid && _myNSrc > 0) {
          const gfx1250_TDM_GROUP1 _pg1 = TdmShape<TokT>(_n);
          if (warpId >= 4) {
            int _tileJ = 0;
            for (int _j = 0; _j < topk; ++_j) {
              if (_srcPtrs[_j] == nullptr) continue;
              TdmIssueLoad<TokT>(_cPullTiles + (size_t)_tileJ * _dynTileElems, _srcPtrs[_j] + _off,
                                 _pg1);
              ++_tileJ;
            }
          }
        }
      }

      // ---- Sequential reduce: one token at a time, all warps ----
      for (int _w = 0; _w < warpNum; ++_w) {
        if (_batchOff + _w >= _tokCount) break;
        const int _wTokenId = _tokStart + _batchOff + _w;
        const int _wNSrc = _warpNSrcLds[_w];

        // Warp _w waits for its TDM loads
        if (warpId == _w) __builtin_amdgcn_s_wait_tensorcnt(0);
        __syncthreads();

        T* _wOutPtr = reinterpret_cast<T*>(args.outTokenBuf) + (size_t)_wTokenId * hiddenDim + _off;

        if (_tdmOk && _wNSrc > 0) {
          TokT* _wTiles =
              reinterpret_cast<TokT*>(sharedMem) + (size_t)_w * _dynSrcMax * _dynTileElems;
          const int _nRed = _wNSrc;
          const int _rowStride = _dynTileElems;

#define _CROW_DEAD(_j) false
          constexpr int _cRedSrcMax = 4;
          constexpr int _cOutVB = 16;
          constexpr int _cV = _cOutVB / (int)sizeof(T);
          constexpr int _cVB = _cV * (int)sizeof(TokT);
          using _CVecT = typename core::VecTypeSelector<_cVB>::dataType;
          using _COutVecT = typename core::VecTypeSelector<_cOutVB>::dataType;
          const bool _cVecOk = ((hiddenDim % (size_t)_cV) == 0) && ((_off % (size_t)_cV) == 0) &&
                               ((_rowStride % _cV) == 0);
          const int _nv = _cVecOk ? (_n / (WS * _cV)) * (WS * _cV) : 0;
          constexpr bool _cFoldMix =
              std::is_same_v<TokT, hip_bfloat16> && ((_cV % 2) == 0) && (_cVB == _cV * 2);

          [[maybe_unused]] int _zRow[_cRedSrcMax];
          [[maybe_unused]] float _zMul[_cRedSrcMax];
          if constexpr (_cFoldMix) {
            int _z0 = 0;
#pragma unroll
            for (int _j = _cRedSrcMax - 1; _j >= 0; --_j)
              if (_j < _nRed && !_CROW_DEAD(_j)) _z0 = _j;
#pragma unroll
            for (int _j = 0; _j < _cRedSrcMax; ++_j) {
              const bool _live = (_j < _nRed) && !_CROW_DEAD(_j);
              _zRow[_j] = _live ? _j : _z0;
              _zMul[_j] = _live ? 1.0f : 0.0f;
            }
          }

          // 8-warp interleaved vectorized reduce from warp _w's tiles
          for (int _e = (warpId * WS + laneId) * _cV; _e < _nv; _e += warpNum * WS * _cV) {
            float _a[_cV];
#pragma unroll
            for (int _k = 0; _k < _cV; ++_k) _a[_k] = 0.0f;

            auto _cFoldRow = [&](int _j, const _CVecT& _sv, float _cMul) {
              if constexpr (_cFoldMix) {
                const uint32_t* _sd = reinterpret_cast<const uint32_t*>(&_sv);
#pragma unroll
                for (int _k = 0; _k < _cV / 2; ++_k) {
                  _a[2 * _k] = MoriFmaMixBf16M<false>(_sd[_k], _cMul, _a[2 * _k]);
                  _a[2 * _k + 1] = MoriFmaMixBf16M<true>(_sd[_k], _cMul, _a[2 * _k + 1]);
                }
              } else {
                (void)_cMul;
#pragma unroll
                for (int _k = 0; _k < _cV; ++_k)
                  _a[_k] += (float)(reinterpret_cast<const TokT*>(&_sv)[_k]);
              }
            };

#define _CROW_AT(_j) (*reinterpret_cast<const _CVecT*>(_wTiles + (size_t)(_j) * _rowStride + _e))

            if (_nRed <= _cRedSrcMax) {
              _CVecT _svR[_cRedSrcMax];
              if constexpr (_cFoldMix) {
#pragma unroll
                for (int _j = 0; _j < _cRedSrcMax; ++_j) _svR[_j] = _CROW_AT(_zRow[_j]);
#pragma unroll
                for (int _j = 0; _j < _cRedSrcMax; ++_j) _cFoldRow(_j, _svR[_j], _zMul[_j]);
              } else {
#pragma unroll
                for (int _j = 0; _j < _cRedSrcMax; ++_j) {
                  _svR[_j] = _CROW_AT((_j < _nRed) ? _j : 0);
                }
#pragma unroll
                for (int _j = 0; _j < _cRedSrcMax; ++_j) {
                  if (_j >= _nRed || _CROW_DEAD(_j)) continue;
                  _cFoldRow(_j, _svR[_j], 1.0f);
                }
              }
            } else {
              for (int _j = 0; _j < _nRed; ++_j) {
                if (_CROW_DEAD(_j)) continue;
                _cFoldRow(_j, _CROW_AT(_j), 1.0f);
              }
            }
#undef _CROW_AT

            union {
              _COutVecT _ov;
              T _oe[_cV];
              uint32_t _op[_cOutVB / 4];
            };
            constexpr bool _cCvtPk =
                std::is_same_v<T, hip_bfloat16> && ((_cV % 2) == 0) && ((_cOutVB / 4) == (_cV / 2));
            if constexpr (_cCvtPk) {
#pragma unroll
              for (int _k = 0; _k < _cV / 2; ++_k)
                _op[_k] = MoriPackTo2<T>(_a[2 * _k], _a[2 * _k + 1]);
            } else {
#pragma unroll
              for (int _k = 0; _k < _cV; ++_k) _oe[_k] = T(_a[_k]);
            }
            static_assert(_cOutVB == 16, "the b128 store is written for the 16 B output vector");
            __builtin_nontemporal_store(*reinterpret_cast<const _mori_v4i*>(&_ov),
                                        reinterpret_cast<_mori_v4i*>(_wOutPtr + _e));
          }

          for (int _e = _nv + warpId * WS + laneId; _e < _n; _e += warpNum * WS) {
            float _acc = 0.0f;
            for (int _j = 0; _j < _nRed; ++_j) {
              if (_CROW_DEAD(_j)) continue;
              _acc += (float)_wTiles[(size_t)_j * _rowStride + _e];
            }
            _wOutPtr[_e] = T(_acc);
          }
#undef _CROW_DEAD
        } else if (_wNSrc > 0 && warpId == _w) {
          // Scalar fallback — owning warp only
          for (int _e = laneId; _e < _n; _e += WS) {
            float _acc = 0.0f;
            for (int _j = 0; _j < topk; ++_j) {
              if (_srcPtrs[_j] == nullptr) continue;
              _acc += (float)(_srcPtrs[_j][_off + _e]);
            }
            _wOutPtr[_e] = T(_acc);
          }
        }

        __syncthreads();
      }
    }

    // Weights — each warp handles its own token
    if constexpr (kCfg.useWeights) {
      if (_myValid && args.outWeightsBuf) {
        core::WarpAccum<float, 4>(args.outWeightsBuf + _myTokenId * topk, _srcWtPtrs, nullptr, topk,
                                  topk);
      }
    }
  }
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
