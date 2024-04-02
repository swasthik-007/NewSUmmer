function fetchdata(url){
    return new Promise(function (resolve,reject){
        console.log("start downloading from ",url);
        setTimeout(function processDownloading(){
            let data="dummy data";
            console.log("donwload completed");
            resolve(data);
        },7000);
    });
}

console.log("start");
let Promiseobj=fetchdata("ssdfsdffrg");
Promiseobj.then(function A(value){
    console.log("value is",value);
})
console.log("end");