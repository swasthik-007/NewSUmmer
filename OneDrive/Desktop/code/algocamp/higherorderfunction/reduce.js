/**
 * HOF available for arrays
 * takes f as  an arguement
 * goes to every element
 * say arr[i]
 * reduce will pass this element to the fn, and accumulate 
 * the result of the further function calls with this particular result
 */

const arr=[ 1,2,3,4,5,6];

function sum(prev,curr){
    // console.log(prev+curr);
    return prev +curr;
}
const result =arr.reduce(sum);
// console.log(result);
/**
 * real life application is adding to cart in 
 */
function addprices(prevresult,currvalue){
    let newprice = prevresult.price +currvalue.price;
return {price:newprice}
}

let cart=[ 
    {
        price:10,name:"iphone"
    },
    {
price:500,name:"backcover"
    }
]
const totalprice= cart.reduce(addprices);
console.log(totalprice.price);