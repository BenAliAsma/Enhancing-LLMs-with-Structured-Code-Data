# Number of parameters to methods
A method (or constructor) that uses a high number of formal parameters makes maintenance more difficult:

* It is difficult to write a call to the method, because the programmer must know how to supply an appropriate value for each parameter.
* It is *externally* difficult to understand, because calls to the method are longer than a single line of code.
* It can be *internally* difficult to understand, because it has so many dependencies.

## Recommendation
Restrict the number of formal parameters for a method, according to the reason for the high number:

* Several of the parameters are logically related, but are passed into the method separately. The parameters that are logically related should be grouped together (see the 'Introduce Parameter Object' refactoring on pp. 238-242 of \[Fowler\]).
* The method has too many responsibilities. It should be broken into multiple methods (see the 'Extract Method' refactoring on pp. 89-95 of \[Fowler\]), and each new method should be passed a subset of the original parameters.
* The method has redundant parameters that are not used. The two main reasons for this are: (1) parameters were added for future extensibility but are never used; (2) the body of the method was changed so that it no longer uses certain parameters, but the method signature was not correspondingly updated. In both cases, the theoretically correct solution is to delete the unused parameters (see the 'Remove Parameter' refactoring on pp. 223-225 of \[Fowler\]), although you must do this cautiously if the method is part of a published interface.
When a method is part of a published interface, one possible solution is to add a new, wrapper method to the interface that has a tidier signature. Alternatively, you can publish a new version of the interface that has a better design. Clearly, however, neither of these solutions is ideal, so you should take care to design interfaces the right way from the start.

The practice of adding parameters for future extensibility is especially bad. It is confusing to other programmers, who are uncertain what values they should pass in for these unnecessary parameters, and it adds unused code that is potentially difficult to remove later.


## Examples
In the following example, although the parameters are logically related, they are passed into the `printAnnotation` method separately.


```java
void printAnnotation(String annotationMessage, int annotationLine, int annotationOffset,
                     int annotationLength) {
    System.out.println("Message: " + annotationMessage);
    System.out.println("Line: " + annotationLine);
    System.out.println("Offset: " + annotationOffset);
    System.out.println("Length: " + annotationLength);
```
In the following modified example, the parameters that are logically related are grouped together in a class, and an instance of the class is passed into the method instead.


```java
class Annotation {
    //...
}

void printAnnotation(Annotation annotation) {
    System.out.println("Message: " + annotation.getMessage());
    System.out.println("Line: " + annotation.getLine());
    System.out.println("Offset: " + annotation.getOffset());
    System.out.println("Length: " + annotation.getLength());
}
```
In the following example, the `printMembership` method has too many responsibilities, and so needs to be passed four arguments.


```java
void printMembership(Set<Fellow> fellows, Set<Member> members, 
                     Set<Associate> associates, Set<Student> students) {
    for(Fellow f: fellows) {
        System.out.println(f);
    }
    for(Member m: members) {
        System.out.println(m);
    }
    for(Associate a: associates) {
        System.out.println(a);
    }
    for(Student s: students) {
        System.out.println(s);
    }
}

void printRecords() {
    //...
    printMembership(fellows, members, associates, students);
}
```
In the following modified example, `printMembership` has been broken into four methods. (For brevity, only one method is shown.) As a result, each new method needs to be passed only one of the original four arguments.


```java
void printFellows(Set<Fellow> fellows) {
    for(Fellow f: fellows) {
        System.out.println(f);
    }
}

//...

void printRecords() {
    //...
    printFellows(fellows);
    printMembers(members);
    printAssociates(associates);
    printStudents(students);
}
```

## References
* M. Fowler, *Refactoring*. Addison-Wesley, 1999.
