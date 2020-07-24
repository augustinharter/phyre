#include <iostream>
#include <string>
using namespace std;

void func(int &arg) {
  cout << to_string(arg);
}

int main() {
  int v = 10;
  int* pt = &v;

  func(v);
  return 0;
}
