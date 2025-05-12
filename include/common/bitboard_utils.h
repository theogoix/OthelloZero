#ifndef BITBOARDUTILS_H
#define BITBOARDUTILS_H

#include<cstdint>

namespace BitboardUtils {

using bitboard = uint64_t;

inline int popcount(bitboard bb){
    return __builtin_popcountll(bb);
};

inline int lsbIndex(bitboard bb){
    return bb ? __builtin_ctzll(bb) : -1;
}

inline bitboard 

}
#endif