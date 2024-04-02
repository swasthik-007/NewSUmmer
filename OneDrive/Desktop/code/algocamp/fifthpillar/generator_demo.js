function* fetchnextelement(){
    console.log("i am inside the generator fn");
    yield 1;
    yield 2;
    console.log("somewhere in the middle");
    yield 3;
    return 10;//return acts as the final yield
    
    yield 4;

}
 const iter=fetchnextelement();
//execution o f generator function doesnt start by calling it
//it returns u an iterator

 console.log(iter.next());
 console.log(iter.next());
 console.log(iter.next());
 console.log(iter.next());

 //yeild is similar to return , but not return
  