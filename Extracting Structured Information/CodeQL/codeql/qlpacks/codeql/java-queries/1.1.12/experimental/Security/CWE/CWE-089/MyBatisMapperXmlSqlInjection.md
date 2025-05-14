# SQL injection in MyBatis Mapper XML
MyBatis allows operating the database by creating XML files to construct dynamic SQL statements. If the syntax `${param}` is used in those statements, and `param` is under the user's control, attackers can exploit this to tamper with the SQL statements or execute arbitrary SQL commands.


## Recommendation
When writing MyBatis mapping statements, try to use the syntax `#{xxx}`. If the syntax `${xxx}` must be used, any parameters included in it should be sanitized to prevent SQL injection attacks.


## Example
The following sample shows several bad and good examples of MyBatis XML files usage. In `bad1`, `bad2`, `bad3`, `bad4`, and `bad5` the syntax `${xxx}` is used to build dynamic SQL statements, which causes a SQL injection vulnerability. In `good1`, the program uses the `${xxx}` syntax, but there are subtle restrictions on the data, while in `good2` the syntax `#{xxx}` is used. In both cases the SQL injection vulnerability is prevented.


```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "http://mybatis.org/dtd/mybatis-3-mapper.dtd" >
<mapper namespace="SqlInjectionMapper">

  <resultMap id="BaseResultMap" type="Test">
    <id column="id" jdbcType="INTEGER" property="id"/>
    <result column="name" jdbcType="VARCHAR" property="name"/>
    <result column="pass" jdbcType="VARCHAR" property="pass"/>
  </resultMap>

  <sql id="Update_By_Example_Where_Clause">
    <where>
      <if test="name != null">
        -- bad
        and name = ${name}
      </if>
      <if test="id != null">
        and id = #{id}
      </if>
    </where>
  </sql>

  <select id="bad1" parameterType="java.lang.String" resultMap="BaseResultMap">
    -- bad
    select id,name from test where name like '%${name}%'
  </select>

  <select id="bad2" parameterType="java.lang.String" resultMap="BaseResultMap">
    -- bad
    select id,name from test order by ${name} desc
  </select>

  <select id="bad3" parameterType="java.lang.String" resultMap="BaseResultMap">
    -- bad
    select id,name from test where name in ${name}
  </select>

  <update id="bad4" parameterType="Test">
    update test
    <set>
      <if test="pass != null">
        pass = #{pass},
      </if>
    </set>
    <if test="_parameter != null">
      -- bad
      <include refid="Update_By_Example_Where_Clause" />
    </if>
  </update>

  <insert id="bad5" parameterType="Test">
    insert into test (name, pass)
    <trim prefix="values (" suffix=")" suffixOverrides=",">
      <if test="name != null">
        -- bad
        name = ${name},
      </if>
      <if test="pass != null">
        -- bad
        pass = ${pass},
      </if>
    </trim>
  </insert>

  <select id="good1" parameterType="java.lang.Integer" resultMap="BaseResultMap">
    -- good
    select id,name from test where id = ${id}
  </select>

  <select id="good2" parameterType="java.lang.String" resultMap="BaseResultMap">
    -- good
    select id,name from test where name = #{name}
  </select>
</mapper>

```

## References
* Fortify: [SQL Injection: MyBatis Mapper](https://vulncat.fortify.com/en/detail?id=desc.config.java.sql_injection_mybatis_mapper).
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).
