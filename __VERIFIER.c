#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern ssize_t read_symbolized(int fd, void *buf, size_t count);

_Bool __VERIFIER_nondet_bool() {
    char x = 0;
    read_symbolized(0, &x, sizeof(x));
    _Bool y = x >=0;
    printf("  <input type=\"bool\">%d</input>\n", y);
    return y;
}

char __VERIFIER_nondet_char() {
    char x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"char\">%d</input>\n", x);
    return x;
}

unsigned char __VERIFIER_nondet_uchar() {
    unsigned char x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned char\">%u</input>\n", x);
    return x;
}

short __VERIFIER_nondet_short() {
    short x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"short\">%hi</input>\n", x);
    return x;
}

unsigned short __VERIFIER_nondet_ushort() {
    unsigned short x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned short\">%hu</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_unsigned_long() {
    unsigned long x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

long __VERIFIER_nondet_long() {
    long x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"long\">%li</input>\n", x);
    return x;
}

unsigned int __VERIFIER_nondet_uint() {
    unsigned int x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned int\">%u</input>\n", x);
    return x;
}

int __VERIFIER_nondet_int() {
    int x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"int\">%d</input>\n", x);
    return x;
}

unsigned __VERIFIER_nondet_unsigned() {
    unsigned x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned\">%d</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_ulong() {
    unsigned long x = 0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

float __VERIFIER_nondet_float() {
    float x = 0.0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"float\">%f</input>\n", x);
    return x;
}

double __VERIFIER_nondet_double() {
    double x = 0.0;
    read_symbolized(0, &x, sizeof(x));
    printf("  <input type=\"double\">%lf</input>\n", x);
    return x;
}
