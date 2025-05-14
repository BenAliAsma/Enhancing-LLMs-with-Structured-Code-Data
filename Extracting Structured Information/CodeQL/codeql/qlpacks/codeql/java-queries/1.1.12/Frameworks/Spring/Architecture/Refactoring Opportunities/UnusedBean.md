# Beans that are never used within the code
Bean definitions that are specified but are never used are redundant and can be removed. Unused beans make the program harder to understand.

A bean definition is considered to be used if one or more of the following is true:

* The bean is referenced or defined in the `<constructor-arg>` or `<property>` element of a live bean.
* The bean is injected in to a constructor or method of a live bean due to autowiring. This includes autowiring by annotation (`@Autowired` or `@Inject`), and autowiring configured by the autowired attribute within bean configuration files.
* The bean is explicitly loaded from a factory bean. It is not always possible to determine when this occurs, because factory beans are loaded using a `String` value, which may contain arbitrary values.
* The bean is called reflectively by the Spring framework. For example, if the class is a Spring MVC framework controller, it may be called in response to web requests.
* The bean has a static initializer.
* The bean is not lazy, and has a constructor or instance initializer that modifies state outside of the bean.
Any bean which is not used in one or more ways will be marked as "dead".


## Recommendation
First verify that the bean definition is never used at runtime. In some cases beans may be used in framework-specific ways, or may be loaded by name from a bean factory in a way that is impossible to determine statically.

After confirming that the bean is not required, remove the bean. You will also need remove any references to this bean, which may, in turn, require removing other beans or references.


## Example
The following example shows a configuration file that includes two beans:


```xml
<beans>
    <!-- This bean is referred to, so is live. -->
    <bean id="petStore" class="org.sample.PetStoreService"/>
    <!-- This bean is never referred to, so is dead. -->
    <bean id="clinic" class="org.sample.ClinicService"/>
</beans>

```
This XML file is loaded with the following Java class:


```java
class Start {
	public static void main(String[] args) {
		// Create a context from the XML file, constructing beans
		ApplicationContext context =
		    new ClassPathXmlApplicationContext(new String[] {"services.xml"});

		// Retrieve the petStore from the context bean factory.
		PetStoreService service = context.getBean("petStore", PetStoreService.class);
		// Use the value
		List<String> userList = service.getUsernameList();
	}
}
```
This class constructs a Spring `ApplicationContext` using the XML file, then loads the "petStore" bean. Given these two files, the "clinic" bean will be marked as dead because it is not used in any context, unlike the "petStore" bean.


## References
* Spring Framework Reference Documentation 4.2: [6.3 Bean overview](http://docs.spring.io/spring/docs/4.2.3.RELEASE/spring-framework-reference/html/beans.html#beans-definition).
