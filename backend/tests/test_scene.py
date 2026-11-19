from app.scene import answer_question, features, infer_relations, iou, make_graph, serialize_graph

def obj(i,label,box): return {"id":f"object-{i}","label":label,"confidence":.9,"bbox":box}
def test_iou(): assert iou([0,0,10,10],[5,5,15,15])==25/175
def test_features(): assert features(obj(1,"cup",[0,0,20,10]),100,50)["normalized_center"]==[.1,.1]
def test_relation_and_no_duplicates():
    rel=infer_relations([obj(1,"cup",[0,0,20,20]),obj(2,"table",[30,50,70,80])],100,100)
    assert any(r["type"]=="above" for r in rel)
    assert len({(r["subject_id"],r["object_id"],r["type"]) for r in rel})==len(rel)
def test_graph_empty_single_and_serialized():
    assert serialize_graph(make_graph([],[],100,100))=={"nodes":[],"edges":[]}
    assert len(serialize_graph(make_graph([obj(1,"cat",[1,2,8,9])],[],10,10))["nodes"])==1
def test_question_answers():
    objs=[obj(1,"cup",[0,0,10,10]),obj(2,"table",[0,30,20,50])]
    rel=infer_relations(objs,100,100)
    assert "cup" in answer_question("What is above the table?",objs,rel)
    assert "1 cup" in answer_question("How many cups?",objs,rel)
