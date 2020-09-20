from neural_net.genetic_algorithm import select_best_parents

parent0 = [-500, "I am", "the", "useless one.."]
parent1 = [-20, "I am", "the", "worse!"]
parent2 = [-10, "I am", "the", "second worse!"]
parent3 = [5, "I am", "the", "second best!"]
parent4 = [15, "I am", "the", "best!"]


def test_select_best_parents():

    parents = [parent1, parent3, parent0, parent4, parent2]
    expected_4 = [parent4[1:], parent3[1:], parent2[1:], parent1[1:]]
    expected_3 = [parent4[1:], parent3[1:], parent2[1:]]
    expected_2 = [parent4[1:], parent3[1:]]
    expected_1 = [parent4[1:]]

    result_4 = select_best_parents(parents, 4)
    result_3 = select_best_parents(parents, 3)
    result_2 = select_best_parents(parents, 2)
    result_1 = select_best_parents(parents, 1)

    assert expected_4 == result_4
    assert expected_3 == result_3
    assert expected_2 == result_2
    assert expected_1 == result_1
