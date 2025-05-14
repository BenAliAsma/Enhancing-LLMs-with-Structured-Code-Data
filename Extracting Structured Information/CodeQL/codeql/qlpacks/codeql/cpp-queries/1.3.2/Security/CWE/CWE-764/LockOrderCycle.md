# Cyclic lock order dependency
Deadlock can occur when two or more threads each attempt to lock multiple mutexes, but not in the same order. A famous example of this is the "Dining Philosophers Problem" (see references below). In this example, deadlock occurs when all five philosophers simultaneously pick up the fork to their left because then none of them can pick up the fork to their right. A simple solution is to assign an order to the mutexes. If the philosophers always pick up the lower numbered fork first, rather than the fork to their left, then deadlock cannot occur.

In general, we can build a graph of the order in which the mutexes are locked. For example, if one of the threads locks mutex A first and B second (while A is still locked), then we add a graph edge from A to B. If the resulting graph contains a cycle then there is a possibility of deadlock. The graph for the dining philosophers example contains a cycle because each philosopher creates a graph edge from their left fork to their right fork.


## Recommendation
If multiple mutexes need to be locked simultaneously, then make sure that all threads lock them in the same order.


## Example
In this example, `f1` locks `mtx1` first and `mtx2` second, but `f2` locks them in the opposite order. If `f1` and `f2` are run simultaneously in separate threads then they could cause a deadlock.


```cpp
std::mutex mtx1;
std::mutex mtx2;

void f1() {
  // GOOD: lock mtx1 before mtx2.
  mtx1.lock();
  mtx2.lock();
  printf("f1");
  mtx2.unlock();
  mtx1.unlock();
}

void f2() {
  // BAD: lock mtx2 before mtx1.
  mtx2.lock();
  mtx1.lock();
  printf("f2");
  mtx1.unlock();
  mtx2.unlock();
}

```

## References
* Dijkstra, E. W., *EWD-310: [Hierarchical Ordering of Sequential Processes](http://www.cs.utexas.edu/users/EWD/ewd03xx/EWD310.PDF)*. E. W. Dijkstra Archive. Center for American History, University of Texas at Austin.
* Hoare, C. A. R., *[Communicating Sequential Processes](http://www.usingcsp.com/cspbook.pdf)*, Section 2.5 - 'Example: The Dining Philosophers,' p. 55, 2004 (originally published in 1985 by Prentice Hall International).
* Common Weakness Enumeration: [CWE-764](https://cwe.mitre.org/data/definitions/764.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
