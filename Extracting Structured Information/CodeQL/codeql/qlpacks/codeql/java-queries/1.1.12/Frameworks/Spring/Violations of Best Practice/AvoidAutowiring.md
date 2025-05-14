# Avoid autowiring
Using Spring autowiring can make it difficult to see what beans get passed to constructors or setters. The Spring Framework Reference documentation cites the following disadvantages of autowiring:

* Explicit dependencies in `property` and `constructor-arg` settings always override autowiring. You cannot autowire so-called *simple* properties such as primitives, `Strings`, and `Classes` (and arrays of such simple properties). This limitation is by design.
* Autowiring is less exact than explicit wiring. Although ... Spring is careful to avoid guessing in case of ambiguity that might have unexpected results, the relationships between your Spring-managed objects are no longer documented explicitly.
* Wiring information may not be available to tools that may generate documentation from a Spring container.
* Multiple bean definitions within the container may match the type specified by the setter method or constructor argument to be autowired. For arrays, collections, or Maps, this is not necessarily a problem. However for dependencies that expect a single value, this ambiguity is not arbitrarily resolved. If no unique bean definition is available, an exception is thrown.

## Recommendation
The Spring Framework Reference documentation suggests the following ways to address problems with autowired beans:

* Abandon autowiring in favor of explicit wiring.
* Avoid autowiring for a bean definition by setting its `autowire-candidate` attributes to `false`.
* Designate a single bean definition as the primary candidate by setting the `primary` attribute of its `<bean/>` element to true.
* If you are using Java 5 or later, implement the more fine-grained control available with annotation-based configuration.

## Example
The following example shows a bean, `autoWiredOrderService`, that is defined using autowiring, and an improved version of the bean, `orderService`, that is defined using explicit wiring.


```xml
<!--AVOID: Using autowiring makes it difficult to see the dependencies of the bean-->
<bean id="autoWiredOrderService"
        class="documentation.examples.spring.OrderService"
        autowire="byName"/>

<!--GOOD: Explicitly specifying the properties of the bean documents its dependencies
    and makes the bean configuration easier to maintain-->
<bean id="orderService"
        class="documentation.examples.spring.OrderService">
        <property name="DAO">
            <idref bean="dao"/>
        </property>
</bean>

```

## References
* Spring Framework Reference Documentation 3.0: [3.4.5.1 Limitations and disadvantages of autowiring](https://docs.spring.io/spring-framework/docs/3.0.x/spring-framework-reference/html/beans.html#beans-autowired-exceptions).
* ONJava: [Twelve Best Practices For Spring XML Configurations](http://onjava.com/pub/a/onjava/2006/01/25/spring-xml-configuration-best-practices.html?page=1).
