/* ========================================================
 *   Copyright (C) 2019 All rights reserved.
 *   
 *   filename : utils.h
 *   author   : ***
 *   date     : 2019-12-02
 *   info     : 
 * ======================================================== */
#ifndef _UTILS_H
#define _UTILS_H

/*
 * MAX limit 
 */
#define MAX_STRING 1024
#define MAX_DOC_LEN 1024

/*
 * type
 */
#define uint32 unsigned int
#define real double

/*
 * ======== argPos ======== 
 * Rarse command argments 
 * 
 * Parameters:
 *   str  - a char array represent a argment name.
 *   argc - number of argments.
 *   argv - a char* array represent argments.
 */
int argPos(char *str, int argc, char **argv);

#endif //UTILS_H
