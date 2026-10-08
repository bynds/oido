#pragma once
#include <stddef.h>
#include <stdint.h>
// SHA-256 of data[0..n) as 64 lowercase hex digits plus NUL.
void sha256_hex(const uint8_t *data, size_t n, char out[65]);
