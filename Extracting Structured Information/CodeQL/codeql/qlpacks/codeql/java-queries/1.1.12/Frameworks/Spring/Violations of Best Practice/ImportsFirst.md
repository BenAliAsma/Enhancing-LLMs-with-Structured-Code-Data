# Imports should come before bean definitions
Putting `import` statements at the top of Spring XML bean definition files is good practice because they give a quick summary of the file's dependencies, and can even be used to document the general architecture of a system.


## Recommendation
Make sure that all `import` statements are at the top of the `<beans>` section of a Spring XML bean definition file.


## Example
The following example shows a `<beans>` section of a Spring XML bean definition file in which an `import` statement is in the middle, and a `<beans>` section in which all the `import` statements are at the top.


```xml
<beans>
    <import resource="services.xml"/>
    
    <bean id="bean1" class="..."/>
    <bean id="bean2" class="..."/>
    
    <!--AVOID: Imports in the middle of a bean configuration make it difficult
        to immediately determine the dependencies of the configuration-->
    <import resource="resources/messageSource.xml"/>

    <bean id="bean3" class="..."/>
    <bean id="bean4" class="..."/>
</beans>


<beans>
    <!--GOOD: Having the imports at the top immediately gives an idea of
        what the dependencies of the configuration are-->
    <import resource="services.xml"/>
    <import resource="resources/messageSource.xml"/>
    
    <bean id="bean1" class="..."/>
    <bean id="bean2" class="..."/>
    <bean id="bean3" class="..."/>
    <bean id="bean4" class="..."/>
</beans>

```

## References
* Spring Framework Reference Documentation 3.0: [3.2.2.1 Composing XML-based configuration metadata](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-factory-xml-import).
