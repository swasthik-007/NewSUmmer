//arrays are also custom objects in js
//index of the elements is the key and the element itself is the value

/**
 * map function
 * map is a hof available with arrays 
 * it takes a function as an arguement -> f
 * it returns an array in which every value is actually populated by calling
 * function f with original array element as arguement

* every element of the original array is passed one by one in the argument function f
 * whatever is the output of each individual element, we populate the output in an array
 * map internaly iterates/loops over every element of the given original array pass that element in the argument functuon f and store the returned value inside an array
 */
function square(el){
    return el*el;//returns square
}
function cube(x){
    return x*x*x;
}
const arr=[1,2,3,4,5];

const res=arr.map(square); //square is function passed as an arguement

const cuberes=arr.map(cube);
// console.log(cuberes);/

//SORTING

let array=[1,10,100,1000,11,12,13,14,9,2,3];
//unsorted array
array.sort();//it sorts the given array // lexographical order 
console.log(array);