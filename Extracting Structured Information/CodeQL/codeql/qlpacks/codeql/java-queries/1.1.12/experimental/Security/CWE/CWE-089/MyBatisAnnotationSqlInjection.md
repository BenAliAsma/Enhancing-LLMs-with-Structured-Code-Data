# SQL injection in MyBatis annotation
MyBatis uses methods with the annotations `@Select`, `@Insert`, etc. to construct dynamic SQL statements. If the syntax `${param}` is used in those statements, and `param` is a parameter of the annotated method, attackers can exploit this to tamper with the SQL statements or execute arbitrary SQL commands.


## Recommendation
When writing MyBatis mapping statements, use the syntax `#{xxx}` whenever possible. If the syntax `${xxx}` must be used, any parameters included in it should be sanitized to prevent SQL injection attacks.


## Example
The following sample shows a bad and a good example of MyBatis annotations usage. The `bad1` method uses `$(name)` in the `@Select` annotation to dynamically build a SQL statement, which causes a SQL injection vulnerability. The `good1` method uses `#{name}` in the `@Select` annotation to dynamically include the parameter in a SQL statement, which causes the MyBatis framework to sanitize the input provided, preventing the vulnerability.


```java
import org.apache.ibatis.annotations.Select;

public interface MyBatisAnnotationSqlInjection {

    @Select("select * from test where name = ${name}")
	public Test bad1(String name);

    @Select("select * from test where name = #{name}")
	public Test good1(String name);
}
```

## References
* Fortify: [SQL Injection: MyBatis Mapper](https://vulncat.fortify.com/en/detail?id=desc.config.java.sql_injection_mybatis_mapper).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).
