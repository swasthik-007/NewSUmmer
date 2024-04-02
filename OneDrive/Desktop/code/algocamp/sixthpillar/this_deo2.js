const iphone={
    name: "iphone",
    price: 10000,
    description:" new one",
    rating : 4.8,
    display:  ()=> {
        let iphonered={
            name:"iphonered",
        price:10000,
        print:()=>{
            console.log(this);
        }
        }
        iphonered.print();
    }
}
iphone.display();