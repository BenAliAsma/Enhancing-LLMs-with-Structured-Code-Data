# Missing setters for property dependency injection
The absence of a matching setter method for a property that is defined in a Spring XML bean causes a validation error when the project is compiled.


## Recommendation
Ensure that there is a setter method in the bean file that matches the property name.


## Example
The following example shows a bean file in which there is no match for the setter method that is in the class.


```xml
<bean id="contentService" class="documentation.examples.spring.ContentService">
	<!--BAD: The setter method in the class is 'setHelper', so this property
	         does not match the setter method.-->
	<property name="transactionHelper">
		<ref bean="transactionHelper"/>
	</property>
</bean>

```
This is the bean class.


```java
// bean class
public class ContentService {
	private TransactionHelper helper;

	// This method does not match the property in the bean file.
	public void setHelper(TransactionHelper helper) {
		this.helper = helper;
	}
}

```
The property `transactionHelper` should instead have the name `helper`.


## References
* Spring Framework Reference Documentation 3.0: [3.4.1.2 Setter-based dependency injection](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-setter-injection).
