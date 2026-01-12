#include <iostream>
#include <vector>
using namespace std;
//예제: 4중 중첩반복문 재귀호출로 대체하기
// 0~n 중 원소 네개를 고르는 모든 경우를 출력하는 문제(순열x조합o?)
void choose(int n, int topick, vector<int>& picked){
    if(topick==0) { 
        for(int num : picked){
            cout<<num<<" ";
        }
    }
        for(int i=0; i<n; i++){
        picked.push_back(i);
        choose(n-1, topick-1, picked);
    }
}