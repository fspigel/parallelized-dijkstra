#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

void printv(double * v, int n){
    for(int i = 0; i < n-1; ++i) printf("%.2f, ", v[i]);
    printf("%.2f\n", v[n-1]);
}

void preprocess(double * E, int n){
    for(int i = 0; i < n*n; ++i){
        if(E[i] == 0) E[i] = INFINITY;
    }
    for(int i = 0; i < n; ++i) E[i*n + i] = 0;
}

int counter(bool * V_t, int n){
    int count = 0;
    for(int i=0; i<n; ++i){
        if (V_t[i]) ++count;
    }
    return count;
}

// Sequential Dijsktra's shortest path algorithm
// Vertices will be denoted as 0 through n-1
// E: double matrix of size 'n*n' - weighted adjacency matrix
// n: number of vertices in the graph
// s: starting node/vertex
// l: a field of length 'n' describing the length of the currently shortest known path from s to a given vetrex
// p: a field of length 'n' describing which adjacent vertex the currently shortest known path goes through
void Dijkstra(double * E, int n, int s, double * l, int * p){
    preprocess(E, n);
    bool V_t[n];

    //preprocessing
    for(int i = 0; i < n; ++i){
        l[i] = E[s*n + i];
        p[i] = s;
        V_t[i] = false;
    }


    V_t[s] = true;

    //iterate until all nodes expanded
    while (counter(V_t, n) < n){
        double dist = INFINITY;
        int u = 0;
        for(int i = 0; i < n; ++i){
            if (V_t[i]) continue;
            if (l[i] < dist){
                dist = l[i];
                u = i;
            }
        }

        printf("Dijkstra expanding node %d\n", u);
        printv(l, n);

        V_t[u] = true;
        for(int i = 0; i < n; ++i){
            if (V_t[i]) continue;
            double dist = l[u] + E[u*n + i];
            if(dist < l[i]){
                l[i] = dist;
                p[i] = u;
            }
        }
    }
}

void main(){
    int n = 5;
    //double * E = (double*)malloc(n*n*sizeof(double));
    double E[25] = {    0, 1, 0, 0, 0, 
                        1, 0, 1, 0, 10, 
                        0, 1, 0, 0, 1,
                        0, 0, 0, 0, 1,
                        0, 10, 1, 1, 0
                    };
    double * l = (double*)malloc(n*sizeof(double));
    int * p = (int*)malloc(n*sizeof(int));
    Dijkstra(E, n, 0, l, p);
    
    printf("RESULTS: \n");
    printv(l, n);

    for(int i = 0; i < n-1; ++i) printf("%d, ", p[i]);
    printf("%d\n", p[n-1]);


}