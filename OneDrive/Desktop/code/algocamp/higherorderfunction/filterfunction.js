/**
 * filter func
 * its a HOF
 * loops over the array elements
 * the arguement f which we have to pass inside filter should always return a boolean, otherwise output would be converted 
 * into a boolean
 */
function oddoreven(x){
    return (x%2==0); //returning a boolean
}

let arr=[1,2,3,4,5,6,7,8,9];

const result =arr.filter(oddoreven);
 console.log(result);