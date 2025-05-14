# Use shortcut forms for values
Shortcut forms, introduced in Spring 1.2, allow nested `value` elements to instead be defined as attributes in the enclosing `property` entry. This leads to shorter XML bean configurations that are easier to read.


## Recommendation
When possible, use the shortcut form for defining bean property values.

Note that this does *not* apply to `idref` elements, which are the preferred form of referring to another bean. These do not have a shortcut form that can still be checked by the XML parser.


## Example
The following example shows how a bean that is defined using shortcut forms is more concise than the same bean defined using nested `value` elements.


```xml
<!--AVOID: Using nested 'value' elements can make the configuration file difficult to read-->
<bean id="serviceRegistry" class="documentation.examples.spring.ServiceRegistry">
	<constructor-arg type="java.lang.String">
		<value>main_service_registry</value>
	</constructor-arg>
	<property name="description">
		<value>Top-level registry for services</value>
	</property>
	<property name="serviceMap">
		<map>
			<entry>
				<key>
					<value>orderService</value>
				</key>
				<value>com.foo.bar.OrderService</value>
			</entry>
			<entry>
				<key>
					<value>billingService</value>
				</key>
				<value>com.foo.bar.BillingService</value>
			</entry>
		</map>
	</property>
</bean>


<!--GOOD: Shortcut forms (Spring 1.2) result in more concise bean definitions-->
<bean id="serviceRegistry" class="documentation.examples.spring.ServiceRegistry">
	<constructor-arg type="java.lang.String" value="main_service_registry"/>
	<property name="description" value="Top-level registry for services"/>
	<property name="serviceMap">
		<map>
			<entry key="orderService" value="com.foo.bar.OrderService"/>
			<entry key="billingService" value="com.foo.bar.BillingService"/>
		</map>
	</property>
</bean>

```

## References
* ONJava: [Twelve Best Practices for Spring XML Configurations](http://www.onjava.com/pub/a/onjava/2006/01/25/spring-xml-configuration-best-practices.html?page=1).
* Spring Framework Reference Documentation 3.0: [3.4.2.1 Straight values (primitives, Strings, and so on)](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-value-element).
