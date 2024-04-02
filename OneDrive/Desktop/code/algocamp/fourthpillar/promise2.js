function getRandomInt(max){
    return Math.floor(Math.random() * max);
}

function createapromisewithtieout() {
    return  new Promise(function executor(resolve,reject){
        console.log("entering the executor callback in the promise constructor");
    setTimeout(function(){
        let num=getRandomInt(10);
        if(num%2==0){
            //if the random is even , we fulfill
            resolve(num);
        }
        else {
            //if odd, we reject
            reject(num);
        }
    },10000);
    console.log("exiting the executor callback in the promise constructor");
    }) ;
}

console.log(" starting.....");
const p=createapromisewithtieout();
console.log("we are now waiting for the promise to complete");
console.log("currently my promise obj is like....",p);
p.then(
    function fulfillhandler(value){console.log("inside the fulfill handler with value",value);},
    function rejectionhandler(value){console.log("inside rejection handler with value",value);}
);