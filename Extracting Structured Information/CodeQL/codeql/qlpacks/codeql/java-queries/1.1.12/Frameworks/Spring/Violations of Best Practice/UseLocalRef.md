# Use local refs when referring to beans in the same file
If at all possible, refer to Spring beans in the same XML file using local references, that is `<idref local="targetBean">`. This requires that the bean being referenced is in the same XML file, and is named using the `id` attribute. Using local references has the advantage of allowing reference errors to be detected during XML parsing, instead of during deployment or instantiation.

From the Spring Framework Reference documentation on `idref` elements:

> \[Using the `idref` tag in a `property` element\] is preferable to \[using the bean name in the property's `value` attribute\], because using the `idref` tag allows the container to validate at deployment time that the referenced, named bean actually exists. In the second variation, no validation is performed on the value that is passed to the \[`name`\] property of the client bean. Typos are only discovered (with most likely fatal results) when the client bean is actually instantiated. If the client bean is a prototype bean, this typo and the resulting exception may only be discovered long after the container is deployed.

Additionally, if the referenced bean is in the same XML unit, and the bean name is the bean `id`, you can use the `local` attribute, which allows the XML parser itself to validate the bean `id` earlier, at XML document parse time.


## Recommendation
Use a local `idref` when referring to beans in the same XML file. This allows errors to be detected earlier, at XML parse time rather than during instantiation.


## Example
In the following example, the `shippingService` bean is shown using the `ref` element, which cannot be checked by the XML parser. The `orderService` bean is shown using the `idref` element, which allows the XML parser to find any errors at parse time.


```xml
<beans>
	<bean id="shippingService" class="documentation.examples.spring.ShippingService">
		<!--AVOID: This form of reference cannot be checked by the XML parser-->
		<property name="dao">
			<ref bean="dao"/>
		</property>
	</bean>
	
	<bean id="orderService" class="documentation.examples.spring.OrderService">
		<!--GOOD: This form of reference allows the XML parser to find any errors at parse time-->
		<property name="dao">
			<idref local="dao"/>
		</property>
	</bean>
	
	<bean id="dao" class="documentation.examples.spring.DAO"/>
</beans>
```

## References
* Spring Framework Reference Documentation 3.0: [3.4.2.1 Straight values (primitives, Strings, and so on)](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-value-element).
