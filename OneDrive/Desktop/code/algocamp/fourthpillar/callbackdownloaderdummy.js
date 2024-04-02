// function download(url,cb){
//     console.log("started downloading from URL",url);
//     setTimeout(function exec(){
//         console.log("completed downloading after 5 s");
//         const content ="ABCDEF";
//         cb(content);
//     },5000);
// }

// download("www.xytz.com",function processDownload(data){
//     console.log("downloaded data is",data);
// })
// we gave the power to someone elses hand


//we can resolve that using promises

function download(url){
    console.log("started downloading content from",url);
    return new Promise(function exec(res,rej){
        setTimeout(function p(){
            console.log("completed downloading data in 5s");
            const content="ABCDEF";
            res(content);
        },5000);
    })
}

download("www.asdas.com")
.then(function fulfillhandler(value){
    console.log("content donwloaded is ",value);
})