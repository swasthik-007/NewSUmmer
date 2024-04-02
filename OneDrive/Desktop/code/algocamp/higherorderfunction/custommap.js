function print(element,idx){
    return `element at index ${idx} is ${element}`;
}
function custommapper(arr,func){
    let result=[];
    for(let i=0;i<arr.length;i++){
        result.push(func(arr[i],i));
    }
    return result;
}
const newarr=[9,8,7,6,5];
const value=custommapper(newarr,print);
// console.log(value);


let b=[1,10,100,1000,11,12,13,14,9,2,3];
//sort b in increasing irder
b.sort(function(a,b){//if a<b ->a-b will be negative ->if cmp function gives negative then a is placed before b

    return a-b;
}) // sort is a HOF ,, the sort func  takes a comparator function as an arguement
 console.log(b);