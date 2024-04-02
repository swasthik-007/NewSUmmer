console.log("start of the file");

setTimeout(function timer1(){
    console.log("Timer 1 is done");
},0);//call back queue

for(let i=0;i<10000000000;i++){
    //something
}

let x=Promise.resolve("sanket's promise"); 
/**
 * the above line gives a already fulfilled promise
 * with 
 * value= sanketspromise
 * and 
 * state=fulfilled
 */

x.then(function processPromise(value){
    console.log("whose Promise ?", value);
});
/**
 * after registering with .then 
 * the onfulfillment array has [processpromise]
 * and
 * the rejection array has [].
 */
setTimeout(function timer2(){
    console.log("timer 2 is done");
},0);///callback queue

console.log("end of the file");

// priority order is microtask>>callback queue 