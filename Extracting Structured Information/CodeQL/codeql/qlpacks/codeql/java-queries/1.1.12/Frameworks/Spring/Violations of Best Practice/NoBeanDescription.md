# This bean does not have a description element
In a Spring XML bean definition file, adding a `<description>` element to a `<bean>` element or the enclosing `<beans>` element to document the purpose of the bean specification is good practice. A `description` element also has the advantage of making it easier for tools to detect and display the documentation for your bean specifications.


## Recommendation
Add a `<description>` element either in the `<bean>` element or its enclosing `<beans>` element.


## Example
The following example shows a Spring XML bean definition file that includes `<description>` elements.


```xml
<beans>
	<!--Using a description element makes it easier for tools to pick up
	    documentation of the bean configuration-->
	<description>
	This file configures the various service beans.
	</description>
	
	<!--You can also put a description element in a bean-->
	<bean id="baseService" abstract="true">
		<description>
		This bean defines base properties common to the service beans
		</description>
		...
	</bean>

	<bean id="shippingService" 
			class="documentation.examples.spring.ShippingService"
			parent="baseService">
		...
	</bean>
	
	<bean id="orderService" 
			class="documentation.examples.spring.OrderService"
			parent="baseService">
		...
	</bean>
</beans>
```

## References
* ONJava: [Twelve Best Practices For Spring XML Configurations](http://onjava.com/pub/a/onjava/2006/01/25/spring-xml-configuration-best-practices.html?page=3).
