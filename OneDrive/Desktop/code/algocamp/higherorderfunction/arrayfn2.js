const newarr=[9,8,7,6,5];
// newarr.sort();
/**
 * if the function that we are passing in map takes two arguements then 
 * first arguement will be accessing the actual value;
 * secind arguement will be accessing index of the value
 */

function print(element,idx){
    return `element at index ${idx} is ${element}`;
}

const result=newarr.map(print);
console.log(result);