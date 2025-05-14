# Beans sharing similar properties
Beans that share a considerable number of similar properties exhibit unnecessary repetition in the bean definitions and make the system's architecture more difficult to see.


## Recommendation
Try to move the properties that the bean definitions share to a common parent bean. This reduces repetition in the bean definitions and gives a clearer picture of the system's architecture.


## Example
The following example shows a configuration file that contains two beans that share several properties with the same values.


```xml
<!--AVOID: 'shippingService' and 'orderService' share several properties with the same values-->
<bean id="shippingService" class="documentation.examples.spring.ShippingService">
	<property name="transactionHelper">
		<ref bean="transactionHelper"/>
	</property>
	<property name="dao">
		<ref bean="dao"/>
	</property>
	<property name="registry">
		<ref bean="basicRegistry"/>
	</property>
	
	<property name="shippingProvider" value="Federal Parcel Service"/>
</bean>

<bean id="orderService" class="documentation.examples.spring.OrderService">
	<property name="transactionHelper">
		<ref bean="transactionHelper"/>
	</property>
	<property name="dao">
		<ref bean="dao"/>
	</property>
	<property name="registry">
		<ref bean="basicRegistry"/>
	</property>
	
	<property name="orderReference" value="8675309"/>
</bean>
```
The following example shows how the shared properties have been moved into a parent bean, `baseService`.


```xml
<!--The 'baseService' bean contains common property definitions for services.-->
<bean id="baseService" abstract="true">
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
		parent="baseService">
	<property name="shippingProvider" value="Federal Parcel Service"/>
</bean>

<bean id="orderService" 
		class="documentation.examples.spring.OrderService"
		parent="baseService">
	<property name="orderReference" value="8675309"/>
</bean>
```

## References
* Spring Framework Reference Documentation 3.0: [3.4.2.2 References to other beans (collaborators)](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-ref-element).
