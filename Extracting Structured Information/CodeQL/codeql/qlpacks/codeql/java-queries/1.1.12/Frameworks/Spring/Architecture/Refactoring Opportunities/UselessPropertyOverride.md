# Useless property override
A property in a child bean that overrides a property with the same name in its parent and has the same contents is useless. This is because the bean inherits the property from its parent anyway.


## Recommendation
If possible, remove the property in the child bean.


## Example
In the following example, `registry` is defined in both the parent bean and the child bean. It should be removed from the child bean.


```xml
<beans>
	<bean id="baseShippingService" abstract="true">
		<property name="transactionHelper">
			<ref bean="transactionHelper"/>
		</property>
		<property name="dao">
			<ref bean="dao"/>
		</property>
		<property name="registry">
			<ref bean="basicRegistry"/>
		</property>
	</bean>

	<bean id="shippingService" 
			class="documentation.examples.spring.ShippingService"
			parent="baseShippingService">
		<!--AVOID: This property is already defined with the same value in the parent bean.-->
		<property name="registry">
			<ref bean="basicRegistry"/>
		</property>
		<property name="shippingProvider" value="Federal Parcel Service"/>
	</bean>
</beans>

```

## References
* Spring Framework Reference Documentation 3.0: [3.7 Bean definition inheritance](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-child-bean-definitions).
