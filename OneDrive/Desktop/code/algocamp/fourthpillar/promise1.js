/**
 * #readability enhancer
 * 
 * # the problem of inversion of control
 * 
 * #in js, promises are special type of object that get 
 * returned immediately when we call them
 * 
 * #promises acts as aplaceholder for the data we hope 
 * to get  back sometime in future
 * 
 * #in these promise objects we can attatch the 
 * functionality we want to execute once the future task is done
 * 
 * #promise objects are native to javascript
 * 
 * #creation of Promise object is sychronous in nature
 * 
 * 
 * #States--->
 * ~Pending=when we create a new prms object this is the default state . (represents work in progress)
 * 
 * ~fulfilled= if the operation is completed succesfully.
 * 
 * ~rejected= if op was not succesfull
 * 
 */

// how to create a  Promise
// new Promise(function(resolve,reject){
//   INSIDE THE FUNCTION WE CAN WRITE OUR TIME CONSUMING TASK
// 
//  })
 
/**
 * WHENEVER IN THE IMPLEMENTATION OF EXECUTOR CALLBACK YOU CALL THE RESOLVE FUNCTION
 * , THE PROMISE GOES TO A FULLFILLED STATE
 * 
 * IF YOU CALL REJECT FUNCTION, IT GOES TO  A REJECTED STATE AND IF YOU DONT CALL ANYTHING 
 * ,PROMISE REMAINS IN PENDING STATE
 * 
 */


