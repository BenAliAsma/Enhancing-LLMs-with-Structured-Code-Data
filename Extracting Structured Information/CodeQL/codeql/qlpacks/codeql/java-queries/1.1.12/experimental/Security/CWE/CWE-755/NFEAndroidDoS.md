# Local Android DoS Caused By NumberFormatException
NumberFormatException (NFE) thrown but not caught by an Android application will crash the application. If the application allows external inputs, an attacker can send an invalid number as intent extra to trigger NFE, which introduces local Denial of Service (Dos) attack.

This is a common problem in Android development since Android components don't have `throw Exception(...)` in their class and method definitions.


## Recommendation
Use the Android methods intended to get number extras e.g. `Intent.getFloatExtra(String name, float defaultValue)` since they have the built-in try/catch processing, or explicitly do try/catch in the application.


## Example
The following example shows both 'BAD' and 'GOOD' configurations. In the 'BAD' configuration, number value is retrieved as string extra then parsed to double. In the 'GOOD' configuration, number value is retrieved as integer extra.


```java
public class NFEAndroidDoS extends Activity {
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_view);

		// BAD: Uncaught NumberFormatException due to remote user inputs
		{
			String minPriceStr = getIntent().getStringExtra("priceMin");
			double minPrice = Double.parseDouble(minPriceStr);	
		}

		// GOOD: Use the proper Android method to get number extra  
		{
			int width = getIntent().getIntExtra("width", 0);
			int height = getIntent().getIntExtra("height", 0);
		}
	}
}
```

## References
* CWE: [CWE-755: Improper Handling of Exceptional Conditions](https://cwe.mitre.org/data/definitions/749.html)
* Android Developers: [Android Crashes](https://developer.android.com/topic/performance/vitals/crash)
* Google Analytics: [Crash and Exception Measurement Using the Google Analytics SDK](https://developers.google.com/analytics/devguides/collection/android/v4/exceptions)
* Common Weakness Enumeration: [CWE-755](https://cwe.mitre.org/data/definitions/755.html).
