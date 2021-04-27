# Swiss Table

A specialized hash table implementation used to dynamically map combinations of key field values to a dense set of integer ids. Ids can later be used in place of keys to identify groups of rows with equal keys.

## Introduction

Hash group-by in Arrow uses a variant of a hash table based on a data structure called Swiss table. Swiss table uses linear probing. There is an array of slots and the information related to inserted keys is stored in these slots. A hash function determines the slot where the search for a matching key will start during hash table lookup. Then the slots are visited sequentially, wrapping around the end of an array, until either a match or an empty slot is found, the latter case meaning that there is no match.  Swiss table organizes the slots in blocks of 8 and has a design that enables data level parallelism at the block level. More precisely, it allows for visiting all slots within a block at once during lookups, by simply using 64-bit arithmetic. SIMD instructions can further enhance this data level parallelism allowing to process multiple blocks related to multiple input keys together using SIMD vectors of 64-bit elements. Occupied slots within a block are always clustered together. The name Swiss table comes from likening resulting sequences of empty slots to holes in a one dimensional cheese.

![alt text](img/key_map.jpg)
