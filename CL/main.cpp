#include <CL/cl.hpp>
#include <iostream>

extern "C" {
#include <opendnn.h>
}

using namespace std;

int factorial(int num) {
    if (num <= 1) {
        return 1;
    } else {
        return num * factorial(num-1);
    }
}


int main () {
    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    opendnnHandle_t temporary;
    opendnnCreate(&temporary);
    return 0;
}

