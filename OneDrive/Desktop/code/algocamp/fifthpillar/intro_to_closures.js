function process(){
    let i=0;
    function innerprocess(){
        i+=1;
        return i;
    }
    return innerprocess;

}

let res=process();
// console.log(res);

console.log("first time",res());
console.log("second time",res());
console.log("third time",res());