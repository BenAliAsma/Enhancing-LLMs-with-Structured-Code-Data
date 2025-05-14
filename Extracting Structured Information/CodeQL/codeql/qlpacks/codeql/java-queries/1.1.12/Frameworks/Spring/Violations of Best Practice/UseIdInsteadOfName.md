# Use id instead of name
To name a Spring bean, it is best to use the `id` attribute instead of the `name` attribute. Using the `id` attribute enables the XML parser to perform additional checks (for example, checking if the `id` in a `ref` attribute is an actual `id` of an XML element).


## Recommendation
Use the `id` attribute instead of the `name` attribute when naming a bean.


## Example
In the following example, the `dao` bean is shown using the `name` attribute, which allows a typo to go undetected because the XML parser does not check `name`. In contrast, using the `id` attribute allows the XML parser to catch the typo.


```xml
<!--AVOID: Using the 'name' attribute disables checking of bean references at XML parse time-->
<bean name="dao" class="documentation.examples.spring.DAO"/>

<bean id="orderService" class="documentation.examples.spring.OrderService">
	<!--The XML parser cannot catch this typo-->
	<property name="dao" ref="da0"/>
</bean>


<!--GOOD: Using the 'id' attribute enables checking of bean references at XML parse time-->
<bean id="dao" class="documentation.examples.spring.DAO"/>

<bean id="orderService" class="documentation.examples.spring.OrderService">
	<!--The XML parser can catch this typo-->
	<property name="dao" ref="da0"/>
</bean>
```

## References
* Spring Framework Reference Documentation 3.0: [3.3.1 Naming beans](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-beanname).
* W3C: [3.3.1 Attribute Types](http://www.w3.org/TR/REC-xml/#sec-attribute-types).
