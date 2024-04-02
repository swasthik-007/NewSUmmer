/**
 * 1. INVERSION OF CONTROL 
 * 2. callback Hell--> readability problem
 */

// let arr=[1,10,1000,9,2,3,11];

// arr.sort(function cmp(a,b){
//     return a-b;
// })

// console.log(arr);

function dotask(fn , x){
    // whole implementation is done b y team a
    fn(x*x); //calling my callback with square of x
}//team a

//here team b tries to use it

dotask(function (num){//due to callbacks iam passing control of how exec should be called to dotask
    //this is called inversion of control
    console.log("the num is",num);

},9);