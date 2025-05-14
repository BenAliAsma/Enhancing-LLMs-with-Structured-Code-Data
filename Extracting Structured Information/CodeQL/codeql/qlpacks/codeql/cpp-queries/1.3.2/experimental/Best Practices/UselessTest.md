# Useless Test
Comparison operations like `a==8 && a!=7` contain a useless part : the non-equal part. This rule finds tests of this kind within an `if` or a `while` statement


## Recommendation
Remove the useless comparisons


## Example

```cpp
void test(){
    int a = 8;
    int b = 9;

    //Useless NonEquals
    if(a==8 && a != 7) {}

    while(a==8 && a!=7){}
}

```
