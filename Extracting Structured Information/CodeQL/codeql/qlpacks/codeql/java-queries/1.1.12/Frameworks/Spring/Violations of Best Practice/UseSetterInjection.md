# Use setter injection instead of constructor injection
When you use the Spring Framework, using setter injection instead of constructor injection is more flexible, particularly for Spring beans with a large number of optional properties. Constructor injection should be used only on required bean properties; using constructor injection on optional bean properties requires a large number of constructors to handle different combinations of properties.

Although the generally accepted best practice is to use constructor injection for mandatory dependencies, and setter injection for optional dependencies, the `@Required` annotation allows you to forgo constructor injection completely. Using the `@Required` annotation on a setter method makes the framework check that a dependency is injected using that method.


## Recommendation
Use setter injection in bean configurations, marking required properties with the `@Required` annotation. It makes it easier to accommodate a large number of optional properties, and makes the bean more flexible by allowing for re-injection of dependencies.


## Example
The following example shows a bean that is defined using constructor injection. The bean configuration is followed by the class definition.


```xml
<!--AVOID: Using constructor args for optional parameters requires one constructor per combination
of properties. This leads to a large number of constructors in the bean class.-->
<bean id="chart1" class="documentation.examples.spring.WrongChartMaker">
	<constructor-arg ref="customTrend"/>
	<constructor-arg ref="customAxis"/>
</bean>
```

```java
// Class for bean 'chart1'
public class WrongChartMaker {
	private AxisRenderer axisRenderer = new DefaultAxisRenderer();
	private TrendRenderer trendRenderer = new DefaultTrendRenderer();
	
	public WrongChartMaker() {}

	// Each combination of the optional parameters must be represented by a constructor.
	public WrongChartMaker(AxisRenderer customAxisRenderer) {
		this.axisRenderer = customAxisRenderer;
	}
	
	public WrongChartMaker(TrendRenderer customTrendRenderer) {
		this.trendRenderer = customTrendRenderer;
	}
	
	public WrongChartMaker(AxisRenderer customAxisRenderer, 
							TrendRenderer customTrendRenderer) {
		this.axisRenderer = customAxisRenderer;
		this.trendRenderer = customTrendRenderer;
	}
}
```
The following example shows how the same bean can be defined using setter injection instead. Again, the bean configuration is followed by the class definition.


```xml
<!--GOOD: Using setter injection requires only one setter for each property.-->
<bean id="chart2" class="documentation.examples.spring.ChartMaker">
	<property name="axisRenderer" ref="customAxis"/>
</bean>
```

```java
// Class for bean 'chart2'
public class ChartMaker {
	private AxisRenderer axisRenderer = new DefaultAxisRenderer();
	private TrendRenderer trendRenderer = new DefaultTrendRenderer();
	
	public ChartMaker() {}
	
	public void setAxisRenderer(AxisRenderer axisRenderer) {
		this.axisRenderer = axisRenderer;
	}
	
	public void setTrendRenderer(TrendRenderer trendRenderer) {
		this.trendRenderer = trendRenderer;
	}
}
```

## References
* Martin Fowler: [Inversion of Control Containers and the Dependency Injection pattern](https://martinfowler.com/articles/injection.html).
* ONJava: [Twelve Best Practices for Spring XML Configurations](http://www.onjava.com/pub/a/onjava/2006/01/25/spring-xml-configuration-best-practices.html?page=3).
* Spring Framework Reference Documentation 3.0: [3.4.1.1 Constructor-based dependency injection](https://docs.spring.io/spring-framework/docs/3.0.x/spring-framework-reference/html/beans.html#beans-constructor-injection), [3.4.1.2 Setter-based dependency injection](https://docs.spring.io/spring-framework/docs/3.0.x/spring-framework-reference/html/beans.html#beans-setter-injection).
* SpringSource: [Setter injection versus constructor injection and the use of @Required](https://spring.io/blog/2007/07/11/setter-injection-versus-constructor-injection-and-the-use-of-required/).
