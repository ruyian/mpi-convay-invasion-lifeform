#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define MASTER 0

/***********************************************************
Helper functions
***********************************************************/

//For exiting on error condition
void die(int lineNo);

//For tracking execution
long long wallClockTime();


/***********************************************************
Square matrix related functions, used by both world and pattern
***********************************************************/

char** allocate_mat(int size, char defaultValue);
void print_mat(char**, int size);
void free_mat(char**);


/***********************************************************
World  related functions
***********************************************************/

#define ALIVE 'X' 
#define DEAD 'O'

char** read_world(char* fname, int* size);
int count_neighbor(char** world, int row, int col);
void evolve_world(char** curWorld, char** nextWorld, int size, int thID, int place);


/***********************************************************
Simple circular linked list for match records
***********************************************************/

typedef struct _match {
    int iteration, row, col, rotation;
    struct _match* next;
} MATCH;


typedef struct {
    int nItem;
    MATCH* tail;
} MATCHLIST;

MATCHLIST* newList();

void deleteList(MATCHLIST*);
void purgeList(MATCHLIST*);
void insertEnd(MATCHLIST*, int, int, int, int);
void printList(MATCHLIST*);


/***********************************************************
Search related functions
***********************************************************/

//Using the compass direction to indicate the rotation of pattern
#define N 0 //0° clockwise
#define E 1 //90° clockwise
#define S 2 //180° clockwise
#define W 3 //90° anti-clockwise

char** read_pattern(char* fname, int* size);

void rotate(char** current, char** rotated, int size);

void search_allPattern(char** world, int wSize, int newSize, int thID, int iteration,
    char** patterns[4], int pSize, MATCHLIST* list);

void search_singlePattern(char** world, int wSize, int newSize, int thID, int interation,
    char** pattern, int pSize, int rotation, MATCHLIST* list);

/*
* Main function of simulation
* It takes in 3 command line arguments: <world file> <iterations> <pattern file>
* 
* ‐ <world file>: Starts with a number N and followed by N rows of N characters 
* representing the initial lifeforms in the world. 'X' and 'O' (alphabet Oh) is used to
* represent live and dead cell respectively.
* 
* ‐ <iterations>: Number of iterations (generations) of evolution, including the initial
* world (i.e. generation 0 is counted).
* 
* ‐ <pattern file>: Format is similar to <world file>. Contains a number M and
* followed by M lines of M characters each. Specify the lifeform pattern to be searched
* for
*/


int main(int argc, char** argv) {

    char** curW, ** nextW, ** temp, dummy[20];
    char** patterns[4];
    int dir, iterations, i;
    int size, patternSize;
    int tag = 1;
    int matches = 0;
    long long before, after;
    MATCHLIST* list;

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <world file> <Iterations> <pattern file>\n", argv[0]);
        exit(1);
    }

    ////////////// MPI stuff ///////////////////////
    int my_rank, new_rank, orig_proc, nProc, myID, rc, i, count, up = 5, down = 6, rotation;
    MPI_Group orig_group, new_group;
    MPI_Comm cartComm, newComm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &orig_proc);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Status Stat;
    MPI_Status stats[12];
    MPI_Request reqs[12];

    // Divide processors into 4 groups, one for every pattern
    nProc = orig_proc / 4;
    int leftovers = orig_proc % 4;
    int new_P[4] = { nProc, nProc, nProc, nProc };
    for (i = 0; i < leftovers; i++)
        new_P[i]++;

    int group3[new_P[3]], group2[new_P[2]], group1[new_P[1]], group0[new_P[0]];

    // Distribute the processors
    for (i = 0; i < nProc; i++) {
        group0[i] = i * 4;
        group1[i] = (i * 4) + 1;
        group2[i] = (i * 4) + 2;
        group3[i] = (i * 4) + 3;
    }
    // Take care of leftovers, could probably be done in a nicer way
    if (leftovers == 3) {
        group0[i] = i * 4;
        group1[i] = (i * 4) + 1;
        group2[i] = (i * 4) + 2;
    }
    else if (leftovers == 2) {
        group0[i] = i * 4;
        group1[i] = (i * 4) + 1;
    }
    else if (leftovers == 1) {
        group0[i] = i * 4;
    }

    // Form the new groups
    if (my_rank % 4 == 0) {
        MPI_Group_incl(orig_group, new_P[0], group0, &new_group);
        rotation = N;
    }
    else if (my_rank % 4 == 1) {
        MPI_Group_incl(orig_group, new_P[1], group1, &new_group);
        rotation = E;
    }
    else if (my_rank % 4 == 2) {
        MPI_Group_incl(orig_group, new_P[2], group2, &new_group);
        rotation = S;
    }
    else if (my_rank % 4 == 3) {
        MPI_Group_incl(orig_group, new_P[3], group3, &new_group);
        rotation = W;
    }

    // Rearrange new communicators as 2D grid, divide in rows only
    int nbrs[2], dim[2] = { nProc, 1 }, periods[2] = { 0,0 }, reorder = 0, coords[2];

    // Configure MPI communicators
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &newComm);
    MPI_Comm_size(newComm, &nProc); // Get number of processors in the current group
    MPI_Cart_create(newComm, 2, dim, periods, reorder, &cartComm);
    MPI_Comm_rank(cartComm, &myID);
    MPI_Cart_coords(cartComm, myID, 2, coords);
    MPI_Cart_shift(cartComm, 0, 1, &nbrs[0], &nbrs[1]); // Shift vertically one step from up to down


    // Only master/root reads the files & input
    if (my_rank == MASTER) {
        //printf("Process %d is master\n", myID);

        curW = read_world(argv[1], &size);
        printf("World Size = %d\n", size);

        iterations = atoi(argv[2]);
        printf("Iterations = %d\n", iterations);

        patterns[N] = read_pattern(argv[3], &patternSize);
        printf("Pattern size = %d\n", patternSize);

        nextW = allocate_mat(size + 2, DEAD);

        for (dir = E; dir <= W; dir++)
        {
            patterns[dir] = allocate_mat(patternSize, DEAD);
            rotate(patterns[dir - 1], patterns[dir], patternSize);
        }

        //Start timer
        before = wallClockTime();

    }

    /**************OLD COMMUNICATOR ******************/

    // Broadcast important information to ALL processes, use the old communicator group
    MPI_Bcast(&size, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&patternSize, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Define sending matrix sizes
    int wMatSize = (size + 2) * (size + 2);           // World sending size
    int pMatSize = patternSize * patternSize;     // Pattern sending size
    int newSize = size / nProc;                   // Rows per block
    int subMatSize = (newSize + 1) * (size + 2) + (patternSize - 1) * (size + 2); // SubWorld sending size
    int updatePMat = (patternSize - 1) * (size + 2);	// Size for updating neighboring world

    MPI_Request updateReq[4];
    MPI_Status updateStat[4];


    // Send correct patterns to master nodes in the communicators, non-blocking
    if (my_rank == MASTER) {
        for (dir = 1; dir < 4; dir++) {
            MPI_Isend(patterns[dir][0], pMatSize, MPI_CHAR, dir, dir, MPI_COMM_WORLD, &reqs[dir - 1]);
            MPI_Isend(curW[0], wMatSize, MPI_CHAR, dir, dir * 2, MPI_COMM_WORLD, &reqs[dir + 2]);
        }
    }
    else if (my_rank != MASTER) {
        // Allocate space on all nodes
        patterns[N] = allocate_mat(patternSize, DEAD);
        // It would be better to just allocate the size needed but I use the existing function "square matrix"
        curW = allocate_mat(size + 2, DEAD);
        nextW = allocate_mat(size + 2, DEAD);

        if (my_rank < 4) {
            MPI_Irecv(patterns[N][0], pMatSize, MPI_CHAR, MASTER, my_rank, MPI_COMM_WORLD, &reqs[my_rank + 5]);
            MPI_Irecv(curW[0], wMatSize, MPI_CHAR, MASTER, my_rank * 2, MPI_COMM_WORLD, &reqs[my_rank + 8]);
        }
    }
    // Wait for all data transfers to complete before using new communicators
    MPI_Barrier(cartComm);


    /************************* NEW COMMUNICATORS ************************/

    // Master nodes broadcast correct pattern to all nodes within new communicators
    MPI_Bcast(patterns[N][0], pMatSize, MPI_CHAR, MASTER, cartComm);


    // Distribute correct initial data to nodes within comminicators, blocking
    for (i = 1; i < nProc; i++) {
        if (i == nProc - 1) // Take care of special case with last node. I don't send the last row because we don't need to update it.
            subMatSize = (newSize + 1) * (size + 2) + (size % nProc) * (size + 2);

        if (myID == MASTER) { //Start one before and end pSize-1 after to get all values needed
            rc = MPI_Send(curW[0 + (i * newSize)], subMatSize, MPI_CHAR, i, tag, cartComm);
        }
        else if (myID == i) {
            rc = MPI_Recv(curW[0 + (i * newSize)], subMatSize, MPI_CHAR, MASTER, tag, cartComm, &Stat);
        }
        tag = tag++;
    }


    if (my_rank == MASTER) {
        after = wallClockTime();
        printf("Initial send took %1.2f seconds\n", ((float)(after - before)) / 1000000000);
    }

    /************************* WORKER FUNCTION ************************/
    list = newList();
    int listSize;


    for (i = 0; i < iterations; i++) {

        // Each process search for one area for a single pattern
        search_singlePattern(curW, size, newSize, myID, i, patterns[N], patternSize, rotation, list);


        // Send back lists to local master with blocking (in order to ensure ordering)
        if (myID == MASTER) {
            for (i = 1; i < nProc; i++) {
                // Receive size of list
                rc = MPI_Recv(&listSize, 1, MPI_INT, i, i, cartComm, &Stat);
                // Receive all results
                if (listSize > 0) {
                    int combSendList[4 * listSize];
                    MPI_Recv(combSendList, 4 * listSize, MPI_INT, i, i + tag, cartComm, &Stat);
                    for (count = 0; count < listSize; count++) {
                        // appending to received result
                        insertEnd(list, combSendList[count * 4 + 0], 
                            combSendList[count * 4 + 1], combSendList[count * 4 + 2], combSendList[count * 4 + 3]);
                    }
                }

            }

        }
        else { // Send size of list to the master process
            rc = MPI_Send(&(list->nItem), 1, MPI_INT, MASTER, myID, cartComm);
            // Iterate through list and send as one message
            if ((list->nItem) > 0) {
                int combSendList[4 * (list->nItem)];
                MATCH* temp;
                temp = list->tail->next;

                for (i = 0; i < list->nItem; i++, temp = temp->next) {
                    combSendList[i * 4 + 0] = temp->iteration;
                    combSendList[i * 4 + 1] = temp->row;
                    combSendList[i * 4 + 2] = temp->col;
                    combSendList[i * 4 + 3] = temp->rotation;
                }
                rc = MPI_Send(combSendList, 4* (list->nItem), MPI_INT, MASTER, myID + tag, cartComm);

                // Start over
                purgeList(list);
            }

        }

        // Send back to GLOBAL master, also by blocking (for correct order)
        if (my_rank == MASTER) {

            for (i = 1; i < 4; i++) {
                // Receive size of list
                rc = MPI_Recv(&listSize, 1, MPI_INT, i, i, MPI_COMM_WORLD, &Stat);

                // Receive all results
                if (listSize > 0) {
                    int combSendList[listSize * 4];
                    MPI_Recv(combSendList, listSize * 4, MPI_INT, i, i + tag, MPI_COMM_WORLD, &Stat);

                    // Add to list
                    for (count = 0; count < listSize; count++) {
                        insertEnd(list, combSendList[count * 4 + 0], combSendList[count * 4 + 1], combSendList[count * 4 + 2], combSendList[count * 4 + 3]);
                    }
                }

            }
            // Print list for every iteration = saves time
            printList(list);
            matches += list->nItem;
            purgeList(list);

        }
        else if (my_rank < 4) {
            // Send size of list
            rc = MPI_Send(&(list->nItem), 1, MPI_INT, MASTER, my_rank, MPI_COMM_WORLD);

            // Iterate through list and send as one message
            if ((list->nItem) > 0) {

                int combSendList[(list->nItem) * 4];

                MATCH* cur;
                cur = list->tail->next;

                for (i = 0; i < list->nItem; i++, cur = cur->next) {
                    combSendList[i * 4 + 0] = cur->iteration;
                    combSendList[i * 4 + 1] = cur->row;
                    combSendList[i * 4 + 2] = cur->col;
                    combSendList[i * 4 + 3] = cur->rotation;
                }
                rc = MPI_Send(combSendList, (list->nItem) * 4, MPI_INT, MASTER, my_rank + tag, MPI_COMM_WORLD);

                purgeList(list);
            }

        }

        //Generate next generation. All nodes do this for their cells. Same updated in all 4 groups
        evolve_world(curW, nextW, size, myID, newSize);

        // Send the updated (pSize-1) rows at top to the rank above and the bottom-most row to rank below
        MPI_Isend(nextW[1 + (myID * newSize)], updatePMat, MPI_CHAR, nbrs[0], up, cartComm, &updateReq[0]); 	// Send to node above
        MPI_Irecv(nextW[1 + ((myID + 1) * newSize)], updatePMat, MPI_CHAR, nbrs[1], up, cartComm, &updateReq[1]); // Receive from node below
        MPI_Isend(nextW[(myID + 1) * newSize], size + 2, MPI_CHAR, nbrs[1], down, cartComm, &updateReq[2]); 		// Send to node below
        MPI_Irecv(nextW[(myID * newSize)], size + 2, MPI_CHAR, nbrs[0], down, cartComm, &updateReq[3]); 		// Receive from node above

        // Make sure the communication is done before updating current world
        MPI_Waitall(4, updateReq, updateStat);

        //Update the local world
        temp = curW;
        curW = nextW;
        nextW = temp;

    }

    //////////////////////////////////////////////////////////////////

    if (my_rank == MASTER) {
        after = wallClockTime();

        printf("List size = %d\n", matches);
        printf("Parallel SETL took %1.2f seconds\n", ((float)(after - before)) / 1000000000);

        free_mat(patterns[1]);
        free_mat(patterns[2]);
        free_mat(patterns[3]);
    }

    MPI_Finalize();

    //Clean up
    deleteList(list);
    free_mat(curW);
    free_mat(nextW);
    free_mat(patterns[0]);

    return 0;
}

/***********************************************************
Helper functions
***********************************************************/


void die(int lineNo) {
    fprintf(stderr, "Error at line %d. Exiting\n", lineNo);
    exit(1);
}

long long wallClockTime() {
#ifdef __linux__
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/***********************************************************
Square matrix related functions, used by both world and pattern
***********************************************************/

char** allocate_mat(int size, char defaultValue) {

    char* contiguous;
    char** matrix;
    int i;

    //Using a least compiler version dependent approach here
    //C99, C11 have a nicer syntax.    
    contiguous = (char*)malloc(sizeof(char) * size * size);
    if (contiguous == NULL)
        die(__LINE__);


    memset(contiguous, defaultValue, size * size);

    //Point the row array to the right place
    matrix = (char**)malloc(sizeof(char*) * size);
    if (matrix == NULL)
        die(__LINE__);

    matrix[0] = contiguous;
    for (i = 1; i < size; i++) {
        matrix[i] = &contiguous[i * size];
    }

    return matrix;
}

void print_mat(char** matrix, int size) {
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf("%c", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void free_mat(char** matrix) {

    if (matrix == NULL) return;

    free(matrix[0]);
}

/***********************************************************
World  related functions
***********************************************************/

char** read_world(char* fname, int* sizePtr) {

    FILE* inf;

    char temp, ** world;
    int i, j;
    int size;

    inf = fopen(fname, "r");
    if (inf == NULL)
        die(__LINE__);


    fscanf(inf, "%d", &size);
    fscanf(inf, "%c", &temp);

    // HALO approach
    // allocated additional top + bottom rows
    // and leftmost and rightmost rows to form a boundary
    // to simplify computation of cell along edges
    world = allocate_mat(size + 2, DEAD);

    for (i = 1; i <= size; i++) {
        for (j = 1; j <= size; j++) {
            fscanf(inf, "%c", &world[i][j]);
        }
        fscanf(inf, "%c", &temp);
    }

    *sizePtr = size;    //return size
    return world;

}

int count_neighbor(char** world, int row, int col) {
    //Assume 1 <= row, col <= size

    int i, j, count;

    count = 0;
    for (i = row - 1; i <= row + 1; i++) {
        for (j = col - 1; j <= col + 1; j++) {
            count += (world[i][j] == ALIVE);
        }
    }
    count -= (world[row][col] == ALIVE); // discount the repeated counting of center element

    return count;

}

void evolve_world(char** curWorld, char** nextWorld, int size, int thID, int place) {

    int i, j, liveNeighbours;
    // Take care of the special case with last node
    int stop = (((thID + 1) * place) + (thID + 1) > size) ? size : (thID * place) + place;

    for (i = 1 + (thID * place); i <= stop; i++) {
        for (j = 1; j <= size; j++) {

            liveNeighbours = count_neighbor(curWorld, i, j);
            nextWorld[i][j] = DEAD;

            //Only take care of alive cases
            if (curWorld[i][j] == ALIVE) {

                if (liveNeighbours == 2 || liveNeighbours == 3)
                    nextWorld[i][j] = ALIVE;

            }
            else if (liveNeighbours == 3)
                nextWorld[i][j] = ALIVE;
        }
    }
}

/***********************************************************
Search related functions
***********************************************************/

char** read_pattern(char* fname, int* sizePtr) {

    FILE* inf;

    char temp, ** pattern;
    int i, j;
    int size;

    inf = fopen(fname, "r");
    if (inf == NULL)
        die(__LINE__);

    fscanf(inf, "%d", &size);
    fscanf(inf, "%c", &temp);

    pattern = allocate_mat(size, DEAD);

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            fscanf(inf, "%c", &pattern[i][j]);
        }
        fscanf(inf, "%c", &temp);
    }

    *sizePtr = size;    //return size
    return pattern;
}


void rotate(char** current, char** rotated, int size) {
    int i, j;
    for (i = 0; i < size; i++) 
        for (j = 0; j < size; j++) 
            rotated[j][size - i - 1] = current[i][j];
}

/* 
* Can be extended to search for multiple patterns. Not in used in the current main().
*/
void search_allPattern(char** world, int wSize, int newSize, int thID, int iteration,
    char** patterns[4], int pSize, MATCHLIST* list) {

    int dir;

    for (dir = N; dir <= W; dir++) {
        search_singlePattern(world, wSize, newSize, thID, iteration,
            patterns[dir], pSize, dir, list);
    }

}

void search_singlePattern(char** world, int wSize, int newSize, int thID, int iteration,
    char** pattern, int pSize, int rotation, MATCHLIST* list) {

    int wRow, wCol, pRow, pCol, match;

    // Take care of last node
    int stop = ((thID + 1) * newSize + (thID + 1) > wSize) ? (wSize - pSize + 1) : (thID + 1) * newSize;


    for (wRow = 1 + (thID * newSize); wRow <= stop; wRow++) {
        for (wCol = 1; wCol <= (wSize - pSize + 1); wCol++) {
            match = 1;
#ifdef DEBUGMORE
            printf("S:(%d, %d)\n", wRow - 1, wCol - 1);
#endif
            for (pRow = 0; match && pRow < pSize; pRow++) {
                for (pCol = 0; match && pCol < pSize; pCol++) {
                    if (world[wRow + pRow][wCol + pCol] != pattern[pRow][pCol]) {
#ifdef DEBUGMORE
                        printf("\tF:(%d, %d) %c != %c\n", pRow, pCol,
                            world[wRow + pRow][wCol + pCol], pattern[pRow][pCol]);
#endif
                        match = 0;
                    }
                }
            }
            if (match) {
                insertEnd(list, iteration, wRow - 1, wCol - 1, rotation);
#ifdef DEBUGMORE
                printf("*** Row = %d, Col = %d\n", wRow - 1, wCol - 1);
#endif
            }
        }
    }
}

/***********************************************************
Linked List helper functions
***********************************************************/

MATCHLIST* newList() {

    MATCHLIST* list;

    list = (MATCHLIST*)malloc(sizeof(MATCHLIST));
    if (list == NULL)
        die(__LINE__);

    list->nItem = 0;
    list->tail = NULL;

    return list;
}

void deleteList(MATCHLIST* list) {

    MATCH* cur, * next;
    int i;

    //delete items first
    if (list->nItem != 0) {
        cur = list->tail->next;
        next = cur->next;
        for (i = 0; i < list->nItem; i++, cur = next, next = next->next) {
            free(cur);
        }
    }
    free(list);
}

void purgeList(MATCHLIST* list) {

    MATCH* cur, * next;
    int i;

    //delete items only
    if (list->nItem != 0) {
        cur = list->tail->next;
        next = cur->next;
        for (i = 0; i < list->nItem; i++, cur = next, next = next->next) {
            free(cur);
        }
    }
    list->nItem = 0;
}

void insertEnd(MATCHLIST* list, int iteration, int row, int col, int rotation) {

    MATCH* newItem;

    newItem = (MATCH*)malloc(sizeof(MATCH));
    if (newItem == NULL)
        die(__LINE__);

    newItem->iteration = iteration;
    newItem->row = row;
    newItem->col = col;
    newItem->rotation = rotation;
    if (list->nItem == 0) {
        newItem->next = newItem;
        list->tail = newItem;
    }
    else {
        newItem->next = list->tail->next;
        list->tail->next = newItem;
        list->tail = newItem;
    }

    (list->nItem)++;

}

void printList(MATCHLIST* list) {

    int i;
    MATCH* cur;

    printf("List size = %d\n", list->nItem);    
    if (list->nItem == 0) return;
    cur = list->tail->next;
    for (i = 0; i < list->nItem; i++, cur = cur->next) {
        printf("%d:%d:%d:%d\n",
            cur->iteration, cur->row, cur->col, cur->rotation);
    }
}

