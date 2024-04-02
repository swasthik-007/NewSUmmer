#include <iostream>
#include <vector>
using namespace std;

void generate_checkerboard(int n) {
    vector<vector<char>> checkerboard(2 * n, vector<char>(2 * n));

    for (int i = 0; i < 2 * n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            checkerboard[i][j] = ((i / 2 + j / 2) % 2 == 0) ? '#' : '.';
        }
    }

    for (int i = 0; i < 2 * n; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            cout << checkerboard[i][j];
        }
        cout << endl;
    }
}

int main() {
    int t;
    cin >> t;

    for (int i = 0; i < t; ++i) {
        int n;
        cin >> n;
        generate_checkerboard(n);
        // cout << endl;
    }

    return 0;
}
