# Too many beans in file
Too many bean definitions in a single file can make the file difficult to understand and maintain. It is also an indication that the architecture of the system is too tightly coupled and can be refactored.


## Recommendation
Refactor related bean definitions into separate files, and compose them using the `<import/>` element.


## Example
The following example shows a configuration file that imports two other configuration files. These two files were created by refactoring a file that contained too many bean definitions.


```xml
<beans>
    <!--Compose configuration files by using the 'import' element.-->
    <import resource="services.xml"/>
    <import resource="resources/messageSource.xml"/>

    <bean id="bean1" class="..."/>
    <bean id="bean2" class="..."/>
</beans>

```

## References
* Spring Framework Reference Documentation 3.0: [3.2.2.1 Composing XML-based configuration metadata](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-factory-xml-import).
