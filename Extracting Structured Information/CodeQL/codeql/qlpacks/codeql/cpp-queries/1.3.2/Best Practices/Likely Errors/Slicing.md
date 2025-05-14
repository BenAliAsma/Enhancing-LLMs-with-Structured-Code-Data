# Slicing
This query finds assignments of a non-reference instance of a derived type to a variable of the base type where the derived type has more fields than the base. These assignments slice off all the fields added by the derived type, and can cause unexpected state when accessed as the derived type.


## Recommendation
Change the type of the variable at the left-hand side of the assignment to the subclass.


## Example

```cpp
static int idctr = 0;
//Basic connection with id
class Connection {
public:
    int connId;
    virtual void print_info() {
        cout << "id: " << connId << "\n";
    }
    Connection() {
        connId = idctr++;
    }
};

//Adds counters, and an overriding print_info
class MeteredConnection : public Connection {
public:
    int txCtr;
    int rxCtr;
    MeteredConnection() {
        txCtr = 0;
        rxCtr = 0;
    }
    virtual void print_info() {
        cout << "id: " << connId << "\n" << "tx/rx: " << txCtr << "/" << rxCtr << "\n";
    }
};

int main(int argc, char* argv[]) {
    Connection conn;
    MeteredConnection m_conn;

    Connection curr_conn = conn;
    curr_conn.print_info();
    curr_conn = m_conn; //Wrong: Derived MetricConnection assigned to Connection 
                        //variable, will slice off the counters and the overriding print_info
    curr_conn.print_info(); //Will not print the counters.

    Connection* curr_pconn = &conn;
    curr_pconn->print_info();
    curr_pconn = &m_conn; //Correct: Pointer assigned to address of the MetricConnection. 
                          //Counters and virtual functions remain intact.
    curr_pconn->print_info(); //Will call the correct method MeteredConnection::print_info
}

```

## References
* Wikipedia: [Object slicing](http://en.wikipedia.org/wiki/Object_slicing).
* DevX.com: [Slicing in C++](http://www.devx.com/tips/Tip/14570).
