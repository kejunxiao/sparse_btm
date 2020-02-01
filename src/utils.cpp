/* ========================================================
 *   Copyright (C) 2019 All rights reserved.
 *   
 *   filename : utils.cpp
 *   author   : ***
 *   date     : 2019-12-09
 *   info     : 
 * ======================================================== */
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int argPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) 
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) { 
                printf("Argument missing for %s\n", str);
                exit(1);
            }    
            return a;
        }    
    return -1;
}
