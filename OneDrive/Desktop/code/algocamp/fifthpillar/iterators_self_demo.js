function fetchnextelement(array){
    let idx=0;
    function next(){
        if(idx==array.length){
            return undefined;
        }
        const nextelement=array[idx];
        idx++;
        return nextelement;

    }
    return {next};
}


//somewhere we consume it

const automaticfetcher=fetchnextelement([99,11,12,13,0,1,2,3,4]);//inside the atumoatic fethecer vqariable we can store the next() function

console.log(automaticfetcher.next());
console.log(automaticfetcher.next());
console.log(automaticfetcher.next());
console.log(automaticfetcher.next());
// console.log(automaticfetcher());
// console.log(automaticfetcher());
// console.log(automaticfetcher());