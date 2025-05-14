# Use constructor-arg types instead of index
Using type matching instead of index matching in a Spring `constructor-arg` element produces a more readable bean definition and is less vulnerable to being broken by a change to the constructor of the bean's underlying class. Index matching should be used only if type matching is not sufficient to remove ambiguity in the constructor arguments.


## Recommendation
The bean definition's `constructor-arg` elements should use type matching instead of index matching.


## Example
The following example shows a bean, `billingService1`, whose `constructor-arg` elements use index matching, and an improved version of the bean, `billingService2`, whose `constructor-arg` elements use type matching.


```xml
<!--AVOID: Using explicit constructor indices makes the bean configuration
           vulnerable to changes to the constructor-->
<bean id="billingService1" class="documentation.examples.spring.BillingService">
    <constructor-arg index="0" value="John Doe"/>
    <constructor-arg index="1" ref="dao"/>
</bean>

<!--GOOD: Using type matching makes the bean configuration more robust to changes in
    the constructor-->
<bean id="billingService2" class="documentation.examples.spring.BillingService">
    <constructor-arg ref="dao"/>
    <constructor-arg type="java.lang.String" value="Jane Doe"/>
</bean>

```

## References
* Spring Framework Reference Documentation 3.0: [3.4.1.1 Constructor-based dependency injection](https://docs.spring.io/spring-framework/docs/3.0.x/spring-framework-reference/html/beans.html#beans-constructor-injection).
* ONJava: [Twelve Best Practices For Spring XML Configurations](http://onjava.com/pub/a/onjava/2006/01/25/spring-xml-configuration-best-practices.html?page=2).
