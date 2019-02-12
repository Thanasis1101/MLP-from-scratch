function goals = labels2goals( labels, number_of_distinct_labels )

    [number_of_all_labels, ~] = size(labels);
    goals = zeros(number_of_all_labels, number_of_distinct_labels);
    
    for i=1:number_of_all_labels
        goals(i, labels(i) + 1) = 1;
    end

end

