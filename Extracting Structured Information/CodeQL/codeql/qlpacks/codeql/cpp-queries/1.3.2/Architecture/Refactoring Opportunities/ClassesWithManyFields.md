# Classes with too many fields
This rule finds classes with more than 15 instance (i.e., non-`static`) fields. Library classes are not shown. Having too many fields in one class is a sign that the class may lack cohesion (i.e. lack a single purpose). These classes can be split into smaller, more cohesive classes. Alternatively, the related fields can be grouped into `struct`s.


## Recommendation
Classes with many fields may be hard to maintain. They could probably be refactored by breaking them down into smaller classes and using composition.


## Example

```cpp
//This struct contains 30 fields.
struct MyParticle {
  bool isActive;
  int priority;

  float x, y, z;
  float dx, dy, dz;
  float ddx, ddy, ddz;
  bool isCollider;

  int age, maxAge;
  float size1, size2;

  bool hasColor;
  unsigned char r1, g1, b1, a1;
  unsigned char r2, g2, b2, a2;

  class texture *tex;
  float u1, v1, u2, v2;
};

```

## References
* W. Stevens, G. Myers, L. Constantine, *Structured Design*. IBM Systems Journal, 13 (2), 115-139, 1974.
* Microsoft Patterns &amp; Practices Team, *Microsoft Application Architecture Guide (2nd Edition), Chapter 3: Architectural Patterns and Styles.* Microsoft Press, 2009 ([available online](http://msdn.microsoft.com/en-us/library/ee658117.aspx)).
* Wikipedia: [Code refactoring](https://en.wikipedia.org/wiki/Code_refactoring)
