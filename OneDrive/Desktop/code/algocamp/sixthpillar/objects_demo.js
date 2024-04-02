let arr=[1,2,3,4];
console.log(typeof arr);

let obj={
    x:10,
    y:20
};
Object.freeze(obj);//neither we can add nor update
obj.x=20;
obj.y=30;

// console.log(obj);

let obj1={
    x:10,
    y:20
};

Object.seal(obj1);//we cannot addd but we can update
obj1.x=30;
obj1.z=60;
console.log(obj1);